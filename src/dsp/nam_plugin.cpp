/*
 * NAM Audio FX Plugin - Neural Amp Modeler for Move Anything
 *
 * Wraps the NeuralAudio library (MIT, by Mike Oliphant) to run .nam and
 * .aidax neural-network guitar-amp models as a Signal Chain audio effect.
 *
 * Includes built-in cabinet impulse response (IR) convolution for amp-only
 * models. Loads mono WAV files from the cabs/ directory and convolves via
 * direct time-domain overlap-save (cab IRs are short, typically <4096 samples).
 *
 * Dependencies (all header-only / static, permissive licenses):
 *   NeuralAudio  - MIT      - Mike Oliphant
 *   Eigen        - MPL2     - Eigen contributors
 *   RTNeural     - BSD-3    - Jatin Chowdhury
 *   math_approx  - BSD-3    - Jatin Chowdhury
 *   nlohmann/json- MIT      - Niels Lohmann
 *
 * Audio: 44100 Hz, 128 frames/block, stereo interleaved int16 in-place.
 * NAM models are mono - we sum L+R to mono, process, write back to both.
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <dirent.h>
#include <algorithm>
#include <string>
#include <atomic>
#include <pthread.h>

/* NeuralAudio */
#include "NeuralAudio/NeuralModel.h"

/* Move Anything API */
extern "C" {
#include "plugin_api_v1.h"
}

/* ======================================================================== */

#define MAX_MODELS 256
#define MAX_CABS 256
#define MAX_NAME_LEN 128
#define MAX_PATH_LEN 512
#define FRAMES_PER_BLOCK 128
#define MAX_IR_LEN 8192

static const host_api_v1_t *g_host = nullptr;

static void plugin_log(const char *msg) {
    if (g_host && g_host->log) g_host->log(msg);
}

/* ======================================================================== */
/* WAV reader - minimal parser for cab IR files                              */
/* ======================================================================== */

/* Read a mono float IR from a WAV file. Supports PCM16 and float32.
 * Returns number of samples read, or 0 on failure. */
static int load_wav_ir(const char *path, float *out, int max_samples) {
    FILE *f = fopen(path, "rb");
    if (!f) return 0;

    /* RIFF header */
    char riff_id[4];
    uint32_t file_size;
    char wave_id[4];
    if (fread(riff_id, 1, 4, f) != 4 || memcmp(riff_id, "RIFF", 4) != 0) { fclose(f); return 0; }
    if (fread(&file_size, 4, 1, f) != 1) { fclose(f); return 0; }
    if (fread(wave_id, 1, 4, f) != 4 || memcmp(wave_id, "WAVE", 4) != 0) { fclose(f); return 0; }

    uint16_t audio_format = 0, num_channels = 0, bits_per_sample = 0;
    uint32_t sample_rate = 0, data_size = 0;
    bool found_fmt = false, found_data = false;

    /* Parse chunks */
    while (!found_data) {
        char chunk_id[4];
        uint32_t chunk_size;
        if (fread(chunk_id, 1, 4, f) != 4) break;
        if (fread(&chunk_size, 4, 1, f) != 1) break;

        if (memcmp(chunk_id, "fmt ", 4) == 0) {
            if (chunk_size < 16) { fclose(f); return 0; }
            fread(&audio_format, 2, 1, f);
            fread(&num_channels, 2, 1, f);
            fread(&sample_rate, 4, 1, f);
            uint32_t byte_rate; fread(&byte_rate, 4, 1, f);
            uint16_t block_align; fread(&block_align, 2, 1, f);
            fread(&bits_per_sample, 2, 1, f);
            /* Skip any extra fmt bytes */
            if (chunk_size > 16) fseek(f, chunk_size - 16, SEEK_CUR);
            found_fmt = true;
        } else if (memcmp(chunk_id, "data", 4) == 0) {
            data_size = chunk_size;
            found_data = true;
        } else {
            fseek(f, chunk_size, SEEK_CUR);
        }
    }

    if (!found_fmt || !found_data) { fclose(f); return 0; }
    /* Accept PCM (1) or IEEE float (3) */
    if (audio_format != 1 && audio_format != 3) { fclose(f); return 0; }

    int bytes_per_sample = bits_per_sample / 8;
    int total_samples = data_size / (bytes_per_sample * num_channels);
    if (total_samples > max_samples) total_samples = max_samples;

    int read_count = 0;
    for (int i = 0; i < total_samples; i++) {
        float sample = 0.0f;
        if (audio_format == 1 && bits_per_sample == 16) {
            int16_t s; fread(&s, 2, 1, f);
            sample = s / 32768.0f;
        } else if (audio_format == 1 && bits_per_sample == 24) {
            uint8_t b[3]; fread(b, 1, 3, f);
            int32_t s = (b[0] | (b[1] << 8) | (b[2] << 16));
            if (s & 0x800000) s |= 0xFF000000; /* sign extend */
            sample = s / 8388608.0f;
        } else if (audio_format == 1 && bits_per_sample == 32) {
            int32_t s; fread(&s, 4, 1, f);
            sample = s / 2147483648.0f;
        } else if (audio_format == 3 && bits_per_sample == 32) {
            fread(&sample, 4, 1, f);
        } else if (audio_format == 3 && bits_per_sample == 64) {
            double d; fread(&d, 8, 1, f);
            sample = (float)d;
        } else {
            fclose(f); return 0;
        }
        /* Skip extra channels */
        if (num_channels > 1) {
            fseek(f, bytes_per_sample * (num_channels - 1), SEEK_CUR);
        }
        out[read_count++] = sample;
    }

    fclose(f);

    char msg[MAX_PATH_LEN + 128];
    snprintf(msg, sizeof(msg), "NAM: loaded cab IR %s (%d samples, %d ch, %d bit, fmt %d)",
             path, read_count, num_channels, bits_per_sample, audio_format);
    plugin_log(msg);

    return read_count;
}

/* ======================================================================== */
/* Instance                                                                  */
/* ======================================================================== */

typedef struct {
    char module_dir[MAX_PATH_LEN];

    /* Model */
    NeuralAudio::NeuralModel *model;
    std::atomic<NeuralAudio::NeuralModel *> pending_model;  /* set by loader thread */
    std::atomic<bool> loading;
    char model_path[MAX_PATH_LEN];
    char model_name[MAX_NAME_LEN];

    /* Scanned model files */
    int model_count;
    char model_names[MAX_MODELS][MAX_NAME_LEN];
    char model_paths[MAX_MODELS][MAX_PATH_LEN];
    int current_model_index;

    /* Cabinet IR */
    float *cab_ir;       /* IR samples (heap allocated) */
    int cab_ir_len;      /* number of IR samples */
    float *cab_history;  /* circular input buffer for convolution */
    int cab_hist_pos;    /* write position in circular buffer */
    bool cab_bypass;     /* true = skip convolution */
    char cab_name[MAX_NAME_LEN];

    /* Scanned cab files */
    int cab_count;
    char cab_names[MAX_CABS][MAX_NAME_LEN];
    char cab_paths[MAX_CABS][MAX_PATH_LEN];
    int current_cab_index;

    /* Parameters */
    float input_level;   /* 0.0 - 1.0 knob position */
    float output_level;  /* 0.0 - 1.0 knob position */
    float input_gain;    /* linear gain */
    float output_gain;   /* linear gain */

    /* Audio buffers (avoid per-block allocation) */
    float mono_in[FRAMES_PER_BLOCK];
    float mono_out[FRAMES_PER_BLOCK];

} nam_instance_t;

/* ======================================================================== */
/* Helpers                                                                   */
/* ======================================================================== */

/* Map 0-1 knob to dB range (-24 to +12), then to linear gain */
static float knob_to_gain(float knob) {
    float db = -24.0f + knob * 36.0f;  /* 0 -> -24dB, 0.5 -> -6dB, 1.0 -> +12dB */
    return powf(10.0f, db / 20.0f);
}

static inline float clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

/* Strip directory and extension from path to get display name */
static void path_to_name(const char *path, char *name, int name_len) {
    const char *slash = strrchr(path, '/');
    const char *base = slash ? slash + 1 : path;
    const char *dot = strrchr(base, '.');
    int len = dot ? (int)(dot - base) : (int)strlen(base);
    if (len >= name_len) len = name_len - 1;
    memcpy(name, base, len);
    name[len] = '\0';
}

/* Check if filename ends with .nam or .json or .aidax */
static bool is_model_file(const char *name) {
    const char *dot = strrchr(name, '.');
    if (!dot) return false;
    return (strcasecmp(dot, ".nam") == 0 ||
            strcasecmp(dot, ".json") == 0 ||
            strcasecmp(dot, ".aidax") == 0);
}

/* Check if filename ends with .wav or .ir */
static bool is_cab_file(const char *name) {
    const char *dot = strrchr(name, '.');
    if (!dot) return false;
    return (strcasecmp(dot, ".wav") == 0 ||
            strcasecmp(dot, ".ir") == 0);
}

/* Generic directory scanner - populates parallel name/path arrays.
 * Returns file count. */
static int scan_directory(const char *dir_path,
                          char names[][MAX_NAME_LEN],
                          char paths[][MAX_PATH_LEN],
                          int max_entries,
                          bool (*filter)(const char *)) {
    int count = 0;
    DIR *dir = opendir(dir_path);
    if (!dir) return 0;

    struct dirent *entry;
    while ((entry = readdir(dir)) != nullptr && count < max_entries) {
        if (entry->d_name[0] == '.') continue;
        if (!filter(entry->d_name)) continue;

        snprintf(paths[count], MAX_PATH_LEN, "%s/%s", dir_path, entry->d_name);
        path_to_name(entry->d_name, names[count], MAX_NAME_LEN);
        count++;
    }
    closedir(dir);

    /* Sort alphabetically by name */
    for (int i = 0; i < count - 1; i++) {
        for (int j = i + 1; j < count; j++) {
            if (strcasecmp(names[i], names[j]) > 0) {
                char tmp_name[MAX_NAME_LEN], tmp_path[MAX_PATH_LEN];
                memcpy(tmp_name, names[i], MAX_NAME_LEN);
                memcpy(tmp_path, paths[i], MAX_PATH_LEN);
                memcpy(names[i], names[j], MAX_NAME_LEN);
                memcpy(paths[i], paths[j], MAX_PATH_LEN);
                memcpy(names[j], tmp_name, MAX_NAME_LEN);
                memcpy(paths[j], tmp_path, MAX_PATH_LEN);
            }
        }
    }

    return count;
}

/* Scan models directory and populate instance model list */
static void scan_models(nam_instance_t *inst) {
    char models_dir[MAX_PATH_LEN];
    snprintf(models_dir, sizeof(models_dir), "%s/models", inst->module_dir);

    inst->model_count = scan_directory(models_dir,
                                       inst->model_names, inst->model_paths,
                                       MAX_MODELS, is_model_file);

    char msg[128];
    snprintf(msg, sizeof(msg), "NAM: found %d model files", inst->model_count);
    plugin_log(msg);
}

/* Scan cabs directory and populate instance cab list */
static void scan_cabs(nam_instance_t *inst) {
    char cabs_dir[MAX_PATH_LEN];
    snprintf(cabs_dir, sizeof(cabs_dir), "%s/cabs", inst->module_dir);

    inst->cab_count = scan_directory(cabs_dir,
                                     inst->cab_names, inst->cab_paths,
                                     MAX_CABS, is_cab_file);

    char msg[128];
    snprintf(msg, sizeof(msg), "NAM: found %d cab IR files", inst->cab_count);
    plugin_log(msg);
}

/* Load a cab IR from file, replacing any previously loaded IR */
static void load_cab(nam_instance_t *inst, int index) {
    if (index < 0 || index >= inst->cab_count) return;

    float *new_ir = (float *)calloc(MAX_IR_LEN, sizeof(float));
    if (!new_ir) return;

    int ir_len = load_wav_ir(inst->cab_paths[index], new_ir, MAX_IR_LEN);
    if (ir_len <= 0) {
        free(new_ir);
        char msg[MAX_PATH_LEN + 64];
        snprintf(msg, sizeof(msg), "NAM: failed to load cab IR %s", inst->cab_paths[index]);
        plugin_log(msg);
        return;
    }

    /* Replace old IR */
    float *old_ir = inst->cab_ir;
    float *old_hist = inst->cab_history;

    inst->cab_ir = new_ir;
    inst->cab_ir_len = ir_len;
    inst->current_cab_index = index;
    path_to_name(inst->cab_paths[index], inst->cab_name, MAX_NAME_LEN);

    /* Allocate new history buffer for convolution */
    inst->cab_history = (float *)calloc(ir_len + FRAMES_PER_BLOCK, sizeof(float));
    inst->cab_hist_pos = 0;

    free(old_ir);
    free(old_hist);

    char msg[MAX_PATH_LEN + 64];
    snprintf(msg, sizeof(msg), "NAM: loaded cab IR '%s' (%d samples)", inst->cab_name, ir_len);
    plugin_log(msg);
}

/* Apply cab IR convolution in-place using direct time-domain overlap-save.
 * Circular buffer avoids per-block allocation. */
static void apply_cab_ir(nam_instance_t *inst, float *audio, int frames) {
    if (!inst->cab_ir || inst->cab_ir_len <= 0 || !inst->cab_history) return;

    const float *ir = inst->cab_ir;
    const int ir_len = inst->cab_ir_len;
    float *hist = inst->cab_history;
    const int hist_len = ir_len + FRAMES_PER_BLOCK;
    int pos = inst->cab_hist_pos;

    for (int i = 0; i < frames; i++) {
        /* Write input sample into circular history */
        hist[pos] = audio[i];

        /* Convolve: sum of ir[k] * hist[pos-k] for k=0..ir_len-1 */
        float sum = 0.0f;
        int p = pos;
        for (int k = 0; k < ir_len; k++) {
            sum += ir[k] * hist[p];
            if (--p < 0) p = hist_len - 1;
        }

        audio[i] = sum;
        if (++pos >= hist_len) pos = 0;
    }

    inst->cab_hist_pos = pos;
}

/* Background model loader thread */
static void *model_loader_thread(void *arg) {
    nam_instance_t *inst = (nam_instance_t *)arg;

    char msg[MAX_PATH_LEN + 64];
    snprintf(msg, sizeof(msg), "NAM: loading model %s", inst->model_path);
    plugin_log(msg);

    NeuralAudio::NeuralModel *new_model =
        NeuralAudio::NeuralModel::CreateFromFile(inst->model_path);

    if (new_model) {
        snprintf(msg, sizeof(msg), "NAM: model loaded successfully (sample_rate=%.0f)",
                 new_model->GetSampleRate());
        plugin_log(msg);
    } else {
        snprintf(msg, sizeof(msg), "NAM: failed to load model %s", inst->model_path);
        plugin_log(msg);
    }

    inst->pending_model.store(new_model, std::memory_order_release);
    inst->loading.store(false, std::memory_order_release);

    return nullptr;
}

static void load_model_async(nam_instance_t *inst, const char *path) {
    if (inst->loading.load(std::memory_order_acquire)) {
        plugin_log("NAM: already loading a model, skipping");
        return;
    }

    strncpy(inst->model_path, path, MAX_PATH_LEN - 1);
    inst->model_path[MAX_PATH_LEN - 1] = '\0';
    path_to_name(path, inst->model_name, MAX_NAME_LEN);

    inst->loading.store(true, std::memory_order_release);

    pthread_t thread;
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);
    pthread_create(&thread, &attr, model_loader_thread, inst);
    pthread_attr_destroy(&attr);
}

/* ======================================================================== */
/* audio_fx_api_v2 implementation                                            */
/* ======================================================================== */

#define AUDIO_FX_API_VERSION_2 2
#define AUDIO_FX_INIT_V2_SYMBOL "move_audio_fx_init_v2"

typedef struct audio_fx_api_v2 {
    uint32_t api_version;
    void* (*create_instance)(const char *module_dir, const char *config_json);
    void (*destroy_instance)(void *instance);
    void (*process_block)(void *instance, int16_t *audio_inout, int frames);
    void (*set_param)(void *instance, const char *key, const char *val);
    int (*get_param)(void *instance, const char *key, char *buf, int buf_len);
    void (*on_midi)(void *instance, const uint8_t *msg, int len, int source);
} audio_fx_api_v2_t;

typedef audio_fx_api_v2_t* (*audio_fx_init_v2_fn)(const host_api_v1_t *host);

/* --- create_instance --- */
static void* v2_create_instance(const char *module_dir, const char *config_json) {
    (void)config_json;
    plugin_log("NAM: creating instance");

    NeuralAudio::NeuralModel::SetDefaultMaxAudioBufferSize(FRAMES_PER_BLOCK);

    nam_instance_t *inst = (nam_instance_t *)calloc(1, sizeof(nam_instance_t));
    if (!inst) return nullptr;

    strncpy(inst->module_dir, module_dir, MAX_PATH_LEN - 1);
    inst->model = nullptr;
    inst->pending_model.store(nullptr);
    inst->loading.store(false);
    inst->current_model_index = -1;

    /* Cabinet IR defaults */
    inst->cab_ir = nullptr;
    inst->cab_ir_len = 0;
    inst->cab_history = nullptr;
    inst->cab_hist_pos = 0;
    inst->cab_bypass = false;
    inst->cab_name[0] = '\0';
    inst->current_cab_index = -1;

    /* Defaults: input at 0.5 (-6dB), output at 0.5 (-6dB) */
    inst->input_level = 0.5f;
    inst->output_level = 0.5f;
    inst->input_gain = knob_to_gain(0.5f);
    inst->output_gain = knob_to_gain(0.5f);

    /* Scan for model files */
    scan_models(inst);

    /* Scan for cab IR files */
    scan_cabs(inst);

    /* Load first model if available */
    if (inst->model_count > 0) {
        inst->current_model_index = 0;
        load_model_async(inst, inst->model_paths[0]);
    }

    /* Load first cab if available */
    if (inst->cab_count > 0) {
        load_cab(inst, 0);
    }

    return inst;
}

/* --- destroy_instance --- */
static void v2_destroy_instance(void *instance) {
    nam_instance_t *inst = (nam_instance_t *)instance;
    if (!inst) return;

    /* Wait for any pending load */
    while (inst->loading.load(std::memory_order_acquire)) {
        struct timespec ts = {0, 10000000}; /* 10ms */
        nanosleep(&ts, nullptr);
    }

    /* Clean up pending model if never consumed */
    NeuralAudio::NeuralModel *pending = inst->pending_model.load(std::memory_order_acquire);
    if (pending) delete pending;

    if (inst->model) delete inst->model;

    /* Clean up cab IR */
    free(inst->cab_ir);
    free(inst->cab_history);

    free(inst);
    plugin_log("NAM: instance destroyed");
}

/* --- process_block --- */
static void v2_process_block(void *instance, int16_t *audio_inout, int frames) {
    nam_instance_t *inst = (nam_instance_t *)instance;
    if (!inst) return;

    /* Check for newly loaded model (lock-free swap) */
    NeuralAudio::NeuralModel *pending = inst->pending_model.load(std::memory_order_acquire);
    if (pending) {
        NeuralAudio::NeuralModel *old = inst->model;
        inst->model = pending;
        inst->pending_model.store(nullptr, std::memory_order_release);
        if (old) delete old;
    }

    /* No model loaded - pass through */
    if (!inst->model) return;

    int n = (frames > FRAMES_PER_BLOCK) ? FRAMES_PER_BLOCK : frames;

    /* Deinterleave stereo int16 -> mono float */
    float ig = inst->input_gain;
    for (int i = 0; i < n; i++) {
        float l = audio_inout[i * 2]     / 32768.0f;
        float r = audio_inout[i * 2 + 1] / 32768.0f;
        inst->mono_in[i] = (l + r) * 0.5f * ig;
    }

    /* Process through NAM */
    inst->model->Process(inst->mono_in, inst->mono_out, (size_t)n);

    /* Apply cab IR convolution (if loaded and not bypassed) */
    if (!inst->cab_bypass && inst->cab_ir) {
        apply_cab_ir(inst, inst->mono_out, n);
    }

    /* Convert back to stereo int16 */
    float og = inst->output_gain;
    for (int i = 0; i < n; i++) {
        float s = clampf(inst->mono_out[i] * og, -1.0f, 1.0f);
        int16_t sample = (int16_t)(s * 32767.0f);
        audio_inout[i * 2]     = sample;
        audio_inout[i * 2 + 1] = sample;
    }
}

/* Minimal JSON field readers for state restore — encoder writes a flat
 * {"key":value,...} object so strstr + atof/sscanf is enough. */
static int nam_json_get_float(const char *json, const char *key, float *out) {
    if (!json || !key || !out) return -1;
    char search[64];
    int n = snprintf(search, sizeof(search), "\"%s\":", key);
    if (n <= 0 || n >= (int)sizeof(search)) return -1;
    const char *p = strstr(json, search);
    if (!p) return -1;
    p += n;
    while (*p == ' ' || *p == '\t') p++;
    *out = (float)atof(p);
    return 0;
}

static int nam_json_get_int(const char *json, const char *key, int *out) {
    if (!json || !key || !out) return -1;
    char search[64];
    int n = snprintf(search, sizeof(search), "\"%s\":", key);
    if (n <= 0 || n >= (int)sizeof(search)) return -1;
    const char *p = strstr(json, search);
    if (!p) return -1;
    p += n;
    while (*p == ' ' || *p == '\t') p++;
    *out = atoi(p);
    return 0;
}

static int nam_json_get_string(const char *json, const char *key,
                               char *out, int out_len) {
    if (!json || !key || !out || out_len <= 0) return -1;
    char search[64];
    int n = snprintf(search, sizeof(search), "\"%s\":\"", key);
    if (n <= 0 || n >= (int)sizeof(search)) return -1;
    const char *p = strstr(json, search);
    if (!p) return -1;
    p += n;
    int i = 0;
    while (*p && *p != '"' && i < out_len - 1) {
        if (*p == '\\' && p[1]) { out[i++] = p[1]; p += 2; continue; }
        out[i++] = *p++;
    }
    out[i] = '\0';
    return i;
}

/* Look up an entry in a sorted name list (model or cab) by exact name.
 * Returns the index or -1 if not found. */
static int nam_find_name(char names[][MAX_NAME_LEN], int count,
                         const char *target) {
    if (!target || !target[0]) return -1;
    for (int i = 0; i < count; i++) {
        if (strcmp(names[i], target) == 0) return i;
    }
    return -1;
}

/* --- set_param --- */
static void v2_set_param(void *instance, const char *key, const char *val) {
    nam_instance_t *inst = (nam_instance_t *)instance;
    if (!inst || !key || !val) return;

    /* Bulk state restore from get_param("state"). Match by saved name
     * first (survives reordering after rescans), fall back to index.
     * Required for slot autosave round-trip — without this the host
     * silently bails on fx:state empty and drops the slot save. */
    if (strcmp(key, "state") == 0) {
        float f;
        int   i;
        char  name[MAX_NAME_LEN];
        if (nam_json_get_float(val, "input_level", &f) == 0) {
            inst->input_level = clampf(f, 0.0f, 1.0f);
            inst->input_gain  = knob_to_gain(inst->input_level);
        }
        if (nam_json_get_float(val, "output_level", &f) == 0) {
            inst->output_level = clampf(f, 0.0f, 1.0f);
            inst->output_gain  = knob_to_gain(inst->output_level);
        }
        /* Model: prefer name (resilient to scan reorder), fall back to idx. */
        int target_model = -1;
        if (nam_json_get_string(val, "model_name", name, sizeof(name)) > 0)
            target_model = nam_find_name(inst->model_names, inst->model_count, name);
        if (target_model < 0 && nam_json_get_int(val, "model_index", &i) == 0)
            if (i >= 0 && i < inst->model_count) target_model = i;
        if (target_model >= 0 && target_model != inst->current_model_index) {
            inst->current_model_index = target_model;
            load_model_async(inst, inst->model_paths[target_model]);
        }
        /* Cab: same name-first / index-fallback pattern. */
        int target_cab = -1;
        if (nam_json_get_string(val, "cab_name", name, sizeof(name)) > 0)
            target_cab = nam_find_name(inst->cab_names, inst->cab_count, name);
        if (target_cab < 0 && nam_json_get_int(val, "cab_index", &i) == 0)
            if (i >= 0 && i < inst->cab_count) target_cab = i;
        if (target_cab >= 0 && target_cab != inst->current_cab_index)
            load_cab(inst, target_cab);
        if (nam_json_get_int(val, "cab_bypass", &i) == 0)
            inst->cab_bypass = (i != 0);
        return;
    }

    if (strcmp(key, "input_level") == 0) {
        inst->input_level = clampf(atof(val), 0.0f, 1.0f);
        inst->input_gain = knob_to_gain(inst->input_level);
    } else if (strcmp(key, "output_level") == 0) {
        inst->output_level = clampf(atof(val), 0.0f, 1.0f);
        inst->output_gain = knob_to_gain(inst->output_level);
    } else if (strcmp(key, "model_index") == 0) {
        int idx = atoi(val);
        if (idx >= 0 && idx < inst->model_count && idx != inst->current_model_index) {
            inst->current_model_index = idx;
            load_model_async(inst, inst->model_paths[idx]);
        }
    } else if (strcmp(key, "model") == 0) {
        /* Direct path load */
        load_model_async(inst, val);
    } else if (strcmp(key, "cab_index") == 0) {
        int idx = atoi(val);
        if (idx >= 0 && idx < inst->cab_count && idx != inst->current_cab_index) {
            load_cab(inst, idx);
        }
    } else if (strcmp(key, "cab_bypass") == 0) {
        inst->cab_bypass = (atoi(val) != 0);
        char msg[64];
        snprintf(msg, sizeof(msg), "NAM: cab bypass %s", inst->cab_bypass ? "on" : "off");
        plugin_log(msg);
    }
}

/* --- get_param --- */
static int v2_get_param(void *instance, const char *key, char *buf, int buf_len) {
    nam_instance_t *inst = (nam_instance_t *)instance;
    if (!inst || !key || !buf) return -1;

    /* Bulk serialization for slot autosave. Persists both name and index
     * for model/cab so restore can survive directory rescans (name match
     * wins, index is a fallback). */
    if (strcmp(key, "state") == 0) {
        return snprintf(buf, buf_len,
            "{\"input_level\":%.4f,\"output_level\":%.4f,"
            "\"model_index\":%d,\"model_name\":\"%s\","
            "\"cab_index\":%d,\"cab_name\":\"%s\","
            "\"cab_bypass\":%d}",
            inst->input_level, inst->output_level,
            inst->current_model_index, inst->model_name,
            inst->current_cab_index, inst->cab_name,
            inst->cab_bypass ? 1 : 0);
    }

    if (strcmp(key, "input_level") == 0)
        return snprintf(buf, buf_len, "%.2f", inst->input_level);
    if (strcmp(key, "output_level") == 0)
        return snprintf(buf, buf_len, "%.2f", inst->output_level);
    if (strcmp(key, "model_name") == 0)
        return snprintf(buf, buf_len, "%s", inst->model_name[0] ? inst->model_name : "(none)");
    if (strcmp(key, "model_count") == 0)
        return snprintf(buf, buf_len, "%d", inst->model_count);
    if (strcmp(key, "model_index") == 0)
        return snprintf(buf, buf_len, "%d", inst->current_model_index);

    /* Dynamic model list for Shadow UI browser - rescan each time */
    if (strcmp(key, "model_list") == 0) {
        scan_models(inst);

        int written = 0;
        written += snprintf(buf + written, buf_len - written, "[");
        for (int i = 0; i < inst->model_count && written < buf_len - 40; i++) {
            if (i > 0) written += snprintf(buf + written, buf_len - written, ",");
            written += snprintf(buf + written, buf_len - written,
                "{\"label\":\"%s\",\"index\":%d}", inst->model_names[i], i);
        }
        written += snprintf(buf + written, buf_len - written, "]");
        return written;
    }

    if (strcmp(key, "loading") == 0)
        return snprintf(buf, buf_len, "%d", inst->loading.load(std::memory_order_acquire) ? 1 : 0);

    /* Cabinet params */
    if (strcmp(key, "cab_name") == 0)
        return snprintf(buf, buf_len, "%s", inst->cab_name[0] ? inst->cab_name : "(none)");
    if (strcmp(key, "cab_count") == 0)
        return snprintf(buf, buf_len, "%d", inst->cab_count);
    if (strcmp(key, "cab_index") == 0)
        return snprintf(buf, buf_len, "%d", inst->current_cab_index);
    if (strcmp(key, "cab_bypass") == 0)
        return snprintf(buf, buf_len, "%d", inst->cab_bypass ? 1 : 0);

    /* Dynamic cab list for Shadow UI browser - rescan each time */
    if (strcmp(key, "cab_list") == 0) {
        scan_cabs(inst);

        int written = 0;
        written += snprintf(buf + written, buf_len - written, "[");
        for (int i = 0; i < inst->cab_count && written < buf_len - 40; i++) {
            if (i > 0) written += snprintf(buf + written, buf_len - written, ",");
            written += snprintf(buf + written, buf_len - written,
                "{\"label\":\"%s\",\"index\":%d}", inst->cab_names[i], i);
        }
        written += snprintf(buf + written, buf_len - written, "]");
        return written;
    }

    /* ui_hierarchy - returned dynamically so model/cab name is current */
    if (strcmp(key, "ui_hierarchy") == 0) {
        const char *hierarchy = "{"
            "\"modes\":null,"
            "\"levels\":{"
                "\"root\":{"
                    "\"label\":\"NAM\","
                    "\"children\":null,"
                    "\"knobs\":[\"input_level\",\"output_level\"],"
                    "\"params\":["
                        "{\"key\":\"input_level\",\"label\":\"Input\"},"
                        "{\"key\":\"output_level\",\"label\":\"Output\"},"
                        "{\"key\":\"cab_bypass\",\"label\":\"Cab Bypass\"},"
                        "{\"level\":\"models\",\"label\":\"Choose Model\"},"
                        "{\"level\":\"cabs\",\"label\":\"Choose Cabinet\"}"
                    "]"
                "},"
                "\"models\":{"
                    "\"label\":\"Model\","
                    "\"items_param\":\"model_list\","
                    "\"select_param\":\"model_index\","
                    "\"children\":null,"
                    "\"knobs\":[],"
                    "\"params\":[]"
                "},"
                "\"cabs\":{"
                    "\"label\":\"Cabinet\","
                    "\"items_param\":\"cab_list\","
                    "\"select_param\":\"cab_index\","
                    "\"children\":null,"
                    "\"knobs\":[],"
                    "\"params\":[]"
                "}"
            "}"
        "}";
        return snprintf(buf, buf_len, "%s", hierarchy);
    }

    return -1;
}

/* ======================================================================== */
/* Entry point                                                               */
/* ======================================================================== */

static audio_fx_api_v2_t g_fx_api_v2;

extern "C" audio_fx_api_v2_t* move_audio_fx_init_v2(const host_api_v1_t *host) {
    g_host = host;

    memset(&g_fx_api_v2, 0, sizeof(g_fx_api_v2));
    g_fx_api_v2.api_version     = AUDIO_FX_API_VERSION_2;
    g_fx_api_v2.create_instance = v2_create_instance;
    g_fx_api_v2.destroy_instance = v2_destroy_instance;
    g_fx_api_v2.process_block   = v2_process_block;
    g_fx_api_v2.set_param       = v2_set_param;
    g_fx_api_v2.get_param       = v2_get_param;
    g_fx_api_v2.on_midi         = nullptr; /* No MIDI handling needed */

    plugin_log("NAM: audio FX plugin initialized (NeuralAudio by Mike Oliphant)");

    return &g_fx_api_v2;
}
