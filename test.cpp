/*
 * VEC 2.0 integration test — named pipes only
 * Curated by @PsyChip <root@psychip.net>
 *
 * Exercises every 2.0 command and prints latency/throughput/byte metrics.
 *
 * Usage:
 *   test <name> <dim>
 *     name : DB name -> connects to \\.\pipe\vec_<name>
 *     dim  : vector dimension (must match server)
 *
 * Build:
 *   cl /O2 /EHsc test.cpp /Fe:test.exe
 *
 * Spec: PROTOCOL-2.0.md
 */

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

/* ---------- protocol constants (mirror vec.cpp) ---------- */
#define BIN_MAGIC        0xF0
#define PROTOCOL_VERSION 0x02

#define CMD_PUSH      0x01
#define CMD_QUERY     0x02
#define CMD_GET       0x04
#define CMD_UPDATE    0x06
#define CMD_DELETE    0x07
#define CMD_LABEL     0x08
#define CMD_UNDO      0x09
#define CMD_SAVE      0x0A
#define CMD_CLUSTER   0x0D
#define CMD_DISTINCT  0x0E
#define CMD_REPRESENT 0x0F
#define CMD_INFO      0x10
#define CMD_QID       0x11
#define CMD_SET_DATA  0x13
#define CMD_GET_DATA  0x14

#define RESP_OK  0x00
#define RESP_ERR 0x01

#define SHAPE_VECTOR 0x01
#define SHAPE_LABEL  0x02
#define SHAPE_DATA   0x04
#define SHAPE_FULL   (SHAPE_VECTOR | SHAPE_LABEL | SHAPE_DATA)

#define GET_MODE_SINGLE 0x00
#define GET_MODE_BATCH  0x01

#define METRIC_L2     0x00
#define METRIC_COSINE 0x01

#define MAX_DATA_BYTES  102400
#define MAX_LABEL_BYTES 2048

/* ---------- test parameters ---------- */
#define NUM_PUSH        1000
#define BATCH_GET_SIZE  100
#define DATA_PAYLOAD_SZ 4096
#define BENCH_QUERIES   200

/* ---------- transport (named pipes) ---------- */
static HANDLE g_pipe = INVALID_HANDLE_VALUE;

/* ---------- byte counters (for debugging / metrics) ---------- */
static unsigned long long g_bytes_sent = 0;
static unsigned long long g_bytes_recv = 0;
static unsigned long long g_frames_sent = 0;
static unsigned long long g_frames_recv = 0;

static int pipe_connect(const char *name) {
    char path[256];
    snprintf(path, sizeof(path), "\\\\.\\pipe\\vec_%s", name);
    for (int tries = 0; tries < 50; tries++) {
        g_pipe = CreateFileA(path, GENERIC_READ | GENERIC_WRITE, 0, NULL, OPEN_EXISTING, 0, NULL);
        if (g_pipe != INVALID_HANDLE_VALUE) return 0;
        if (GetLastError() != ERROR_PIPE_BUSY && GetLastError() != ERROR_FILE_NOT_FOUND) break;
        Sleep(100);
    }
    fprintf(stderr, "ERROR: cannot open %s (last error %lu)\n", path, GetLastError());
    return -1;
}

static int io_send(const void *data, int len) {
    DWORD written;
    int sent = 0;
    const char *p = (const char *)data;
    while (sent < len) {
        if (!WriteFile(g_pipe, p + sent, len - sent, &written, NULL) || written == 0) return -1;
        sent += (int)written;
    }
    g_bytes_sent += (unsigned long long)len;
    g_frames_sent++;
    return sent;
}

static int io_recv(void *buf, int len) {
    DWORD got;
    int total = 0;
    char *p = (char *)buf;
    while (total < len) {
        if (!ReadFile(g_pipe, p + total, len - total, &got, NULL) || got == 0) return -1;
        total += (int)got;
    }
    return total;
}

/* ---------- frame builder ---------- */
static int build_frame(unsigned char *out, unsigned char cmd,
                       const void *label, int label_len,
                       const void *body, int body_len) {
    unsigned char *p = out;
    *p++ = BIN_MAGIC;
    unsigned short ns_len = 0;
    memcpy(p, &ns_len, 2); p += 2;
    *p++ = cmd;
    unsigned short ll = (unsigned short)label_len;
    memcpy(p, &ll, 2); p += 2;
    if (label_len > 0) { memcpy(p, label, label_len); p += label_len; }
    unsigned int blen = (unsigned int)body_len;
    memcpy(p, &blen, 4); p += 4;
    if (body_len > 0) { memcpy(p, body, body_len); p += body_len; }
    return (int)(p - out);
}

/* ---------- response envelope reader ---------- */
static int recv_response(unsigned char **out_body, unsigned int *out_len) {
    unsigned char hdr[5];
    if (io_recv(hdr, 5) < 0) { *out_body = NULL; *out_len = 0; return -1; }
    unsigned char status = hdr[0];
    unsigned int body_len;
    memcpy(&body_len, hdr + 1, 4);
    unsigned char *body = NULL;
    if (body_len > 0) {
        body = (unsigned char *)malloc(body_len);
        if (!body) { *out_body = NULL; *out_len = 0; return -1; }
        if (io_recv(body, body_len) < 0) { free(body); *out_body = NULL; *out_len = 0; return -1; }
    }
    *out_body = body;
    *out_len = body_len;
    g_bytes_recv += (unsigned long long)(5 + body_len);
    g_frames_recv++;
    return status;
}

/* ---------- timing ---------- */
typedef LARGE_INTEGER tstamp_t;
static LARGE_INTEGER g_freq;
static void now(tstamp_t *t) { QueryPerformanceCounter(t); }
static double ms_since(const tstamp_t *t0) {
    tstamp_t t1; QueryPerformanceCounter(&t1);
    return (double)(t1.QuadPart - t0->QuadPart) * 1000.0 / (double)g_freq.QuadPart;
}

/* ---------- check / passes ---------- */
static int g_pass = 0, g_fail = 0;
static void check(const char *name, int ok, const char *msg) {
    if (ok) { printf("  [PASS] %-32s %s\n", name, msg ? msg : ""); g_pass++; }
    else    { printf("  [FAIL] %-32s %s\n", name, msg ? msg : ""); g_fail++; }
}

static void print_err(const char *what, unsigned char *body, unsigned int blen) {
    if (body && blen > 0) printf("    err on %s: \"%.*s\"\n", what, (int)blen, (const char *)body);
    else                  printf("    err on %s: <empty>\n", what);
}

/* ---------- helpers ---------- */
static float randf() { return (float)rand() / (float)RAND_MAX * 2.0f - 1.0f; }

/* ---------- per-command stats ---------- */
typedef struct {
    const char *name;
    double total_ms;
    double min_ms;
    double max_ms;
    int n;
    unsigned long long bytes_sent;
    unsigned long long bytes_recv;
} stat_t;

#define MAX_STATS 32
static stat_t g_stats[MAX_STATS];
static int g_stat_n = 0;

static stat_t *stat_find(const char *name) {
    for (int i = 0; i < g_stat_n; i++) if (strcmp(g_stats[i].name, name) == 0) return &g_stats[i];
    if (g_stat_n >= MAX_STATS) return NULL;
    g_stats[g_stat_n].name = name;
    g_stats[g_stat_n].total_ms = 0;
    g_stats[g_stat_n].n = 0;
    g_stats[g_stat_n].min_ms = 1e18;
    g_stats[g_stat_n].max_ms = 0;
    g_stats[g_stat_n].bytes_sent = 0;
    g_stats[g_stat_n].bytes_recv = 0;
    return &g_stats[g_stat_n++];
}

/* one round-trip wrapper: send frame, recv response, record stats */
static int rt(const char *stat_name, const unsigned char *frame, int flen,
              unsigned char **out_body, unsigned int *out_len) {
    stat_t *s = stat_find(stat_name);
    unsigned long long s0 = g_bytes_sent, r0 = g_bytes_recv;
    tstamp_t t0; now(&t0);
    if (io_send(frame, flen) < 0) {
        if (out_body) *out_body = NULL;
        if (out_len) *out_len = 0;
        return -1;
    }
    unsigned char *body; unsigned int blen;
    int status = recv_response(&body, &blen);
    double dt = ms_since(&t0);
    if (s) {
        s->total_ms += dt;
        s->n++;
        if (dt < s->min_ms) s->min_ms = dt;
        if (dt > s->max_ms) s->max_ms = dt;
        s->bytes_sent += (g_bytes_sent - s0);
        s->bytes_recv += (g_bytes_recv - r0);
    }
    if (out_body) *out_body = body; else free(body);
    if (out_len) *out_len = blen;
    return status;
}

/* =====================================================================
 * MAIN
 * ===================================================================== */
int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <name> <dim>\n", argv[0]);
        return 1;
    }
    const char *name = argv[1];
    int dim = atoi(argv[2]);
    if (dim <= 0) { fprintf(stderr, "ERROR: bad dim\n"); return 1; }

    QueryPerformanceFrequency(&g_freq);
    srand((unsigned)time(NULL));

    printf("VEC 2.0 integration test\n");
    printf("  db=%s dim=%d transport=named-pipe\n", name, dim);

    if (pipe_connect(name) < 0) return 1;

    /* scratch buffers */
    int vbytes = dim * (int)sizeof(float);
    int max_body = vbytes + 4 + MAX_DATA_BYTES + 16;
    int max_frame = 1 + 2 + 1 + 2 + MAX_LABEL_BYTES + 4 + max_body;
    unsigned char *frame = (unsigned char *)malloc(max_frame);
    unsigned char *bodybuf = (unsigned char *)malloc(max_body);
    if (!frame || !bodybuf) { fprintf(stderr, "OOM\n"); return 1; }

    float *vec  = (float *)malloc(vbytes);
    float *qvec = (float *)malloc(vbytes);
    unsigned char *resp = NULL;
    unsigned int resp_len = 0;
    int status;

    int *pushed_idx = (int *)malloc(NUM_PUSH * sizeof(int));
    for (int i = 0; i < NUM_PUSH; i++) pushed_idx[i] = -1;
    int idx_plain = -1;
    int idx_with_data = -1;
    char first_label[64];
    char data_label[64];
    /* unique-per-run suffix so reruns don't collide on the same DB */
    unsigned int run_tag = (unsigned int)((unsigned long long)time(NULL) ^ ((unsigned long long)GetCurrentProcessId() << 16));
    snprintf(data_label, sizeof(data_label), "test/data-payload-%08x", run_tag);
    printf("  run_tag=%08x (labels: rec/%08x/NNNNN)\n", run_tag, run_tag);
    printf("\n");
    unsigned char *data_blob = (unsigned char *)malloc(DATA_PAYLOAD_SZ);
    for (int i = 0; i < DATA_PAYLOAD_SZ; i++) data_blob[i] = (unsigned char)(i & 0xFF);

    /* ======================================================
     * 1) INFO
     * ====================================================== */
    printf("[1] CMD_INFO\n");
    {
        int flen = build_frame(frame, CMD_INFO, NULL, 0, NULL, 0);
        status = rt("INFO", frame, flen, &resp, &resp_len);
        if (status != RESP_OK) { print_err("INFO", resp, resp_len); free(resp); return 1; }

        int srv_dim, srv_count, srv_deleted;
        memcpy(&srv_dim, resp, 4);
        memcpy(&srv_count, resp + 4, 4);
        memcpy(&srv_deleted, resp + 8, 4);
        unsigned char fmt = resp[12];
        long long mtime; memcpy(&mtime, resp + 13, 8);
        unsigned int crc; memcpy(&crc, resp + 21, 4);
        unsigned char crc_ok = resp[25];
        unsigned int name_len; memcpy(&name_len, resp + 26, 4);
        char srv_name[256] = {0};
        if (name_len > 0 && name_len < 256) memcpy(srv_name, resp + 30, name_len);
        unsigned char proto = resp[30 + name_len];

        printf("    dim=%d count=%d deleted=%d fmt=%s name=%s crc=0x%08X crc_ok=%u proto=0x%02x\n",
               srv_dim, srv_count, srv_deleted, fmt == 1 ? "f16" : "f32", srv_name, crc, crc_ok, proto);

        check("INFO returns OK",         1, NULL);
        check("INFO dim matches",        srv_dim == dim, NULL);
        check("INFO protocol = 2",       proto == PROTOCOL_VERSION, NULL);
        free(resp);
    }
    printf("\n");

    /* ======================================================
     * 2) PUSH
     * ====================================================== */
    printf("[2] CMD_PUSH (1 plain + %d labeled + 1 with data + 2 negative)\n", NUM_PUSH);

    /* (a) push without label or data */
    {
        for (int d = 0; d < dim; d++) vec[d] = randf();
        int blen = vbytes + 4;
        memcpy(bodybuf, vec, vbytes);
        unsigned int zero = 0; memcpy(bodybuf + vbytes, &zero, 4);
        int flen = build_frame(frame, CMD_PUSH, NULL, 0, bodybuf, blen);
        status = rt("PUSH-plain", frame, flen, &resp, &resp_len);
        if (status == RESP_OK && resp_len == 4) {
            memcpy(&idx_plain, resp, 4);
            char m[64]; snprintf(m, sizeof(m), "-> slot %d", idx_plain);
            check("PUSH no label no data", 1, m);
        } else { print_err("PUSH plain", resp, resp_len); check("PUSH no label no data", 0, NULL); }
        free(resp);
    }

    /* (b) NUM_PUSH labeled pushes (no data). URI-style labels with / are allowed. */
    int batch_ok = 0, batch_fail = 0;
    char first_fail_msg[128] = {0};
    {
        tstamp_t batch_t0; now(&batch_t0);
        for (int i = 0; i < NUM_PUSH; i++) {
            for (int d = 0; d < dim; d++) vec[d] = randf();
            char lbl[64];
            snprintf(lbl, sizeof(lbl), "rec/%08x/%05d", run_tag, i);
            if (i == 0) strncpy(first_label, lbl, sizeof(first_label));
            int blen = vbytes + 4;
            memcpy(bodybuf, vec, vbytes);
            unsigned int zero = 0; memcpy(bodybuf + vbytes, &zero, 4);
            int flen = build_frame(frame, CMD_PUSH, lbl, (int)strlen(lbl), bodybuf, blen);
            status = rt("PUSH-labeled", frame, flen, &resp, &resp_len);
            if (status != RESP_OK || resp_len != 4) {
                if (batch_fail == 0 && resp && resp_len > 0) {
                    snprintf(first_fail_msg, sizeof(first_fail_msg),
                             "first fail at i=%d lbl=\"%s\" err=\"%.*s\"",
                             i, lbl, (int)resp_len, (const char *)resp);
                }
                batch_fail++;
                free(resp);
                continue;
            }
            memcpy(&pushed_idx[i], resp, 4);
            free(resp);
            batch_ok++;
        }
        double batch_ms = ms_since(&batch_t0);
        char m[160];
        if (batch_fail > 0) {
            snprintf(m, sizeof(m), "ok=%d fail=%d in %.2f ms (%.0f/s) | %s",
                     batch_ok, batch_fail, batch_ms, batch_ok * 1000.0 / batch_ms, first_fail_msg);
            check("PUSH labeled batch", 0, m);
        } else {
            snprintf(m, sizeof(m), "%d records in %.2f ms (%.0f/s)",
                     batch_ok, batch_ms, batch_ok * 1000.0 / batch_ms);
            check("PUSH labeled batch", 1, m);
        }
    }

    /* (c) push with data (must have label) */
    {
        for (int d = 0; d < dim; d++) vec[d] = randf();
        unsigned int dlen = DATA_PAYLOAD_SZ;
        int blen = vbytes + 4 + (int)dlen;
        memcpy(bodybuf, vec, vbytes);
        memcpy(bodybuf + vbytes, &dlen, 4);
        memcpy(bodybuf + vbytes + 4, data_blob, dlen);
        int flen = build_frame(frame, CMD_PUSH, data_label, (int)strlen(data_label), bodybuf, blen);
        status = rt("PUSH-with-data", frame, flen, &resp, &resp_len);
        if (status == RESP_OK && resp_len == 4) {
            memcpy(&idx_with_data, resp, 4);
            char m[64]; snprintf(m, sizeof(m), "-> slot %d (%u B blob)", idx_with_data, dlen);
            check("PUSH vector+label+data", 1, m);
        } else { print_err("PUSH+data", resp, resp_len); check("PUSH vector+label+data", 0, NULL); }
        free(resp);
    }

    /* (d) PUSH error: data without label */
    {
        for (int d = 0; d < dim; d++) vec[d] = randf();
        unsigned int dlen = 16;
        int blen = vbytes + 4 + (int)dlen;
        memcpy(bodybuf, vec, vbytes);
        memcpy(bodybuf + vbytes, &dlen, 4);
        memset(bodybuf + vbytes + 4, 0xAA, dlen);
        int flen = build_frame(frame, CMD_PUSH, NULL, 0, bodybuf, blen);
        status = rt("PUSH-err-nolabel", frame, flen, &resp, &resp_len);
        char m[96];
        if (status == RESP_ERR && resp_len > 0)
            snprintf(m, sizeof(m), "err=\"%.*s\"", (int)resp_len, (const char *)resp);
        else m[0] = '\0';
        check("PUSH data-without-label rejected", status == RESP_ERR, m);
        free(resp);
    }

    /* (e) PUSH error: oversize label */
    {
        char big_label[MAX_LABEL_BYTES + 16];
        for (int i = 0; i < (int)sizeof(big_label); i++) big_label[i] = 'a';
        for (int d = 0; d < dim; d++) vec[d] = randf();
        int blen = vbytes + 4;
        memcpy(bodybuf, vec, vbytes);
        unsigned int zero = 0; memcpy(bodybuf + vbytes, &zero, 4);
        int flen = build_frame(frame, CMD_PUSH, big_label, MAX_LABEL_BYTES + 5, bodybuf, blen);
        status = rt("PUSH-err-biglabel", frame, flen, &resp, &resp_len);
        char m[96];
        if (status == RESP_ERR && resp_len > 0)
            snprintf(m, sizeof(m), "err=\"%.*s\"", (int)resp_len, (const char *)resp);
        else m[0] = '\0';
        check("PUSH oversize-label rejected", status == RESP_ERR, m);
        free(resp);
    }

    /* fresh random query */
    for (int d = 0; d < dim; d++) qvec[d] = randf();
    printf("\n");

    /* ======================================================
     * 3) QUERY
     * ====================================================== */
    printf("[3] CMD_QUERY\n");
    {
        int blen = 2 + vbytes;
        bodybuf[0] = METRIC_L2;
        bodybuf[1] = SHAPE_FULL;
        memcpy(bodybuf + 2, qvec, vbytes);
        int flen = build_frame(frame, CMD_QUERY, NULL, 0, bodybuf, blen);
        status = rt("QUERY-L2-full", frame, flen, &resp, &resp_len);
        if (status == RESP_OK && resp_len >= 4) {
            unsigned int count; memcpy(&count, resp, 4);
            char m[96]; snprintf(m, sizeof(m), "L2 full top-%u, body=%u B", count, resp_len);
            check("QUERY L2 full", count > 0, m);
        } else { print_err("QUERY L2", resp, resp_len); check("QUERY L2 full", 0, NULL); }
        free(resp);
    }
    {
        int blen = 2 + vbytes;
        bodybuf[0] = METRIC_COSINE;
        bodybuf[1] = SHAPE_LABEL;
        memcpy(bodybuf + 2, qvec, vbytes);
        int flen = build_frame(frame, CMD_QUERY, NULL, 0, bodybuf, blen);
        status = rt("QUERY-cos-lean", frame, flen, &resp, &resp_len);
        char m[64]; snprintf(m, sizeof(m), "body=%u B (vs full)", resp_len);
        check("QUERY cosine lean", status == RESP_OK, m);
        free(resp);
    }
    {
        int blen = 2 + vbytes;
        bodybuf[0] = 0xFF;
        bodybuf[1] = SHAPE_FULL;
        memcpy(bodybuf + 2, qvec, vbytes);
        int flen = build_frame(frame, CMD_QUERY, NULL, 0, bodybuf, blen);
        status = rt("QUERY-err-metric", frame, flen, &resp, &resp_len);
        check("QUERY bad metric rejected", status == RESP_ERR, NULL);
        free(resp);
    }
    printf("\n");

    /* ======================================================
     * 4) QID
     * ====================================================== */
    printf("[4] CMD_QID\n");
    int probe_idx = -1;
    for (int i = NUM_PUSH / 2; i < NUM_PUSH; i++) { if (pushed_idx[i] >= 0) { probe_idx = pushed_idx[i]; break; } }
    if (probe_idx < 0) for (int i = 0; i < NUM_PUSH; i++) { if (pushed_idx[i] >= 0) { probe_idx = pushed_idx[i]; break; } }

    if (probe_idx >= 0) {
        int blen = 2 + 4;
        bodybuf[0] = METRIC_L2;
        bodybuf[1] = SHAPE_LABEL;
        memcpy(bodybuf + 2, &probe_idx, 4);
        int flen = build_frame(frame, CMD_QID, NULL, 0, bodybuf, blen);
        status = rt("QID-by-index", frame, flen, &resp, &resp_len);
        char m[64]; snprintf(m, sizeof(m), "probe slot %d, body=%u B", probe_idx, resp_len);
        check("QID by index L2", status == RESP_OK, m);
        free(resp);
    } else check("QID by index L2", 0, "no usable pushed index");

    if (batch_ok > 0) {
        int blen = 2;
        bodybuf[0] = METRIC_COSINE;
        bodybuf[1] = SHAPE_LABEL;
        int flen = build_frame(frame, CMD_QID, first_label, (int)strlen(first_label), bodybuf, blen);
        status = rt("QID-by-label", frame, flen, &resp, &resp_len);
        char m[96];
        if (status == RESP_ERR && resp_len > 0)
            snprintf(m, sizeof(m), "lbl=\"%s\" err=\"%.*s\"", first_label, (int)resp_len, (const char *)resp);
        else snprintf(m, sizeof(m), "lbl=\"%s\" body=%u B", first_label, resp_len);
        check("QID by label cosine", status == RESP_OK, m);
        free(resp);
    } else check("QID by label cosine", 0, "no labeled push succeeded");
    printf("\n");

    /* ======================================================
     * 5) GET
     * ====================================================== */
    printf("[5] CMD_GET\n");
    if (probe_idx >= 0) {
        int blen = 2 + 4;
        bodybuf[0] = GET_MODE_SINGLE;
        bodybuf[1] = SHAPE_FULL;
        memcpy(bodybuf + 2, &probe_idx, 4);
        int flen = build_frame(frame, CMD_GET, NULL, 0, bodybuf, blen);
        status = rt("GET-single-full", frame, flen, &resp, &resp_len);
        char m[64]; snprintf(m, sizeof(m), "body=%u B", resp_len);
        check("GET single by index full", status == RESP_OK, m);
        free(resp);
    }

    if (batch_ok > 0) {
        int blen = 2;
        bodybuf[0] = GET_MODE_SINGLE;
        bodybuf[1] = SHAPE_LABEL | SHAPE_VECTOR;
        int flen = build_frame(frame, CMD_GET, first_label, (int)strlen(first_label), bodybuf, blen);
        status = rt("GET-single-by-label", frame, flen, &resp, &resp_len);
        char m[96];
        if (status == RESP_ERR && resp_len > 0)
            snprintf(m, sizeof(m), "lbl=\"%s\" err=\"%.*s\"", first_label, (int)resp_len, (const char *)resp);
        else snprintf(m, sizeof(m), "lbl=\"%s\" body=%u B", first_label, resp_len);
        check("GET single by label", status == RESP_OK, m);
        free(resp);
    } else check("GET single by label", 0, "no labeled push succeeded");

    /* batch */
    int batch_n = 0;
    {
        int n = 0;
        unsigned char *bb = (unsigned char *)malloc(2 + 4 + BATCH_GET_SIZE * 4);
        bb[0] = GET_MODE_BATCH;
        bb[1] = SHAPE_VECTOR;
        for (int i = 0; i < NUM_PUSH && n < BATCH_GET_SIZE; i++) {
            if (pushed_idx[i] < 0) continue;
            memcpy(bb + 6 + n * 4, &pushed_idx[i], 4);
            n++;
        }
        unsigned int u = (unsigned int)n;
        memcpy(bb + 2, &u, 4);
        int blen = 2 + 4 + n * 4;
        int flen = build_frame(frame, CMD_GET, NULL, 0, bb, blen);
        free(bb);
        if (n > 0) {
            status = rt("GET-batch", frame, flen, &resp, &resp_len);
            char m[96]; snprintf(m, sizeof(m), "%d ids -> %u B (%.1f B/id)", n, resp_len, n ? (double)resp_len / n : 0.0);
            check("GET batch vector-only", status == RESP_OK, m);
            batch_n = n;
            free(resp);
        } else check("GET batch vector-only", 0, "no usable pushed indices");
    }

    {
        int blen = 2 + 4;
        bodybuf[0] = GET_MODE_SINGLE;
        bodybuf[1] = SHAPE_FULL;
        int bad = -1;
        memcpy(bodybuf + 2, &bad, 4);
        int flen = build_frame(frame, CMD_GET, NULL, 0, bodybuf, blen);
        status = rt("GET-err-bad-idx", frame, flen, &resp, &resp_len);
        check("GET bad index rejected", status == RESP_ERR, NULL);
        free(resp);
    }
    printf("\n");

    /* ======================================================
     * 6) UPDATE
     * ====================================================== */
    printf("[6] CMD_UPDATE\n");
    if (probe_idx >= 0) {
        for (int d = 0; d < dim; d++) vec[d] = randf();
        int blen = 4 + vbytes;
        memcpy(bodybuf, &probe_idx, 4);
        memcpy(bodybuf + 4, vec, vbytes);
        int flen = build_frame(frame, CMD_UPDATE, NULL, 0, bodybuf, blen);
        status = rt("UPDATE", frame, flen, &resp, &resp_len);
        check("UPDATE by index", status == RESP_OK && resp_len == 0, NULL);
        free(resp);
    } else check("UPDATE by index", 0, "no probe idx");
    printf("\n");

    /* ======================================================
     * 7) CMD_LABEL
     * ====================================================== */
    printf("[7] CMD_LABEL\n");
    if (probe_idx >= 0) {
        char new_lbl[64];
        snprintf(new_lbl, sizeof(new_lbl), "rec/relabeled-%08x", run_tag);
        int flen = build_frame(frame, CMD_LABEL, new_lbl, (int)strlen(new_lbl), &probe_idx, 4);
        status = rt("LABEL-set", frame, flen, &resp, &resp_len);
        check("LABEL set new", status == RESP_OK, NULL);
        free(resp);
    } else check("LABEL set new", 0, "no probe idx");

    if (probe_idx >= 0) {
        const char *bad = "has spaces and /slash";
        int flen = build_frame(frame, CMD_LABEL, bad, (int)strlen(bad), &probe_idx, 4);
        status = rt("LABEL-err-chars", frame, flen, &resp, &resp_len);
        check("LABEL invalid chars rejected", status == RESP_ERR, NULL);
        free(resp);
    }
    printf("\n");

    /* ======================================================
     * 8) SET_DATA / GET_DATA
     * ====================================================== */
    printf("[8] CMD_SET_DATA / CMD_GET_DATA\n");
    if (idx_with_data >= 0) {
        unsigned int dlen = 256;
        unsigned char tiny[256];
        for (unsigned int i = 0; i < dlen; i++) tiny[i] = (unsigned char)(0x55 ^ (i & 0xFF));

        int blen = 8 + dlen;
        memcpy(bodybuf, &idx_with_data, 4);
        memcpy(bodybuf + 4, &dlen, 4);
        memcpy(bodybuf + 8, tiny, dlen);
        int flen = build_frame(frame, CMD_SET_DATA, NULL, 0, bodybuf, blen);
        status = rt("SET_DATA", frame, flen, &resp, &resp_len);
        check("SET_DATA replace", status == RESP_OK, NULL);
        free(resp);

        flen = build_frame(frame, CMD_GET_DATA, NULL, 0, &idx_with_data, 4);
        status = rt("GET_DATA", frame, flen, &resp, &resp_len);
        if (status == RESP_OK && resp_len >= 4) {
            unsigned int got_len; memcpy(&got_len, resp, 4);
            int matches = (got_len == dlen) && memcmp(resp + 4, tiny, dlen) == 0;
            char m[64]; snprintf(m, sizeof(m), "got %u B, content match=%d", got_len, matches);
            check("GET_DATA roundtrip matches", matches, m);
        } else { print_err("GET_DATA", resp, resp_len); check("GET_DATA roundtrip matches", 0, NULL); }
        free(resp);

        unsigned int huge = MAX_DATA_BYTES + 100;
        unsigned char *big = (unsigned char *)malloc(huge);
        memset(big, 0xCC, huge);
        blen = 8 + huge;
        unsigned char *bb = (unsigned char *)malloc(blen);
        memcpy(bb, &idx_with_data, 4);
        memcpy(bb + 4, &huge, 4);
        memcpy(bb + 8, big, huge);
        flen = build_frame(frame, CMD_SET_DATA, NULL, 0, bb, blen);
        free(bb); free(big);
        status = rt("SET_DATA-err-big", frame, flen, &resp, &resp_len);
        check("SET_DATA oversize rejected", status == RESP_ERR, NULL);
        free(resp);
    } else {
        check("SET_DATA replace", 0, "no idx_with_data");
        check("GET_DATA roundtrip matches", 0, "no idx_with_data");
        check("SET_DATA oversize rejected", 0, "no idx_with_data");
    }
    printf("\n");

    /* ======================================================
     * 9) CLUSTER / DISTINCT / REPRESENT
     * ====================================================== */
    printf("[9] CMD_CLUSTER / CMD_DISTINCT / CMD_REPRESENT\n");
    {
        unsigned char body[9];
        float eps = 0.5f;
        int min_pts = 2;
        memcpy(body, &eps, 4);
        body[4] = METRIC_L2;
        memcpy(body + 5, &min_pts, 4);
        int flen = build_frame(frame, CMD_CLUSTER, NULL, 0, body, 9);
        status = rt("CLUSTER", frame, flen, &resp, &resp_len);
        char m[64]; snprintf(m, sizeof(m), "body=%u B", resp_len);
        check("CLUSTER ok", status == RESP_OK, m);
        free(resp);
    }
    {
        unsigned char body[5];
        int k = 16;
        memcpy(body, &k, 4);
        body[4] = METRIC_L2;
        int flen = build_frame(frame, CMD_DISTINCT, NULL, 0, body, 5);
        status = rt("DISTINCT", frame, flen, &resp, &resp_len);
        char m[64]; snprintf(m, sizeof(m), "status=%s body=%u B", status == RESP_OK ? "OK" : "ERR", resp_len);
        check("DISTINCT responded", status >= 0, m);
        free(resp);
    }
    {
        unsigned char body[9];
        float eps = 0.5f;
        int min_pts = 2;
        memcpy(body, &eps, 4);
        body[4] = METRIC_L2;
        memcpy(body + 5, &min_pts, 4);
        int flen = build_frame(frame, CMD_REPRESENT, NULL, 0, body, 9);
        status = rt("REPRESENT", frame, flen, &resp, &resp_len);
        char m[64]; snprintf(m, sizeof(m), "status=%s body=%u B", status == RESP_OK ? "OK" : "ERR", resp_len);
        check("REPRESENT responded", status >= 0, m);
        free(resp);
    }
    printf("\n");

    /* ======================================================
     * 10) DELETE
     * ====================================================== */
    printf("[10] CMD_DELETE\n");
    if (idx_with_data >= 0) {
        int idx = idx_with_data;
        int flen = build_frame(frame, CMD_DELETE, NULL, 0, &idx, 4);
        status = rt("DELETE", frame, flen, &resp, &resp_len);
        check("DELETE record with data", status == RESP_OK, NULL);
        free(resp);

        flen = build_frame(frame, CMD_DELETE, NULL, 0, &idx, 4);
        status = rt("DELETE-err-twice", frame, flen, &resp, &resp_len);
        check("DELETE already-deleted rejected", status == RESP_ERR, NULL);
        free(resp);

        flen = build_frame(frame, CMD_GET_DATA, NULL, 0, &idx, 4);
        status = rt("GET_DATA-err-deleted", frame, flen, &resp, &resp_len);
        check("GET_DATA on deleted rejected", status == RESP_ERR, NULL);
        free(resp);
    } else {
        check("DELETE record with data", 0, "no idx_with_data");
        check("DELETE already-deleted rejected", 0, "no idx_with_data");
        check("GET_DATA on deleted rejected", 0, "no idx_with_data");
    }
    printf("\n");

    /* ======================================================
     * 11) UNDO
     * ====================================================== */
    printf("[11] CMD_UNDO\n");
    {
        int flen = build_frame(frame, CMD_UNDO, NULL, 0, NULL, 0);
        status = rt("UNDO", frame, flen, &resp, &resp_len);
        check("UNDO", status == RESP_OK, NULL);
        free(resp);
    }
    printf("\n");

    /* ======================================================
     * 12) SAVE
     * ====================================================== */
    printf("[12] CMD_SAVE\n");
    {
        int flen = build_frame(frame, CMD_SAVE, NULL, 0, NULL, 0);
        status = rt("SAVE", frame, flen, &resp, &resp_len);
        if (status == RESP_OK && resp_len >= 8) {
            unsigned int saved, crc;
            memcpy(&saved, resp, 4);
            memcpy(&crc, resp + 4, 4);
            char m[96]; snprintf(m, sizeof(m), "saved=%u crc=0x%08X", saved, crc);
            check("SAVE", 1, m);
        } else { print_err("SAVE", resp, resp_len); check("SAVE", 0, NULL); }
        free(resp);
    }
    printf("\n");

    /* ======================================================
     * 13) Unknown command
     * ====================================================== */
    printf("[13] unknown command\n");
    {
        int flen = build_frame(frame, 0xEE, NULL, 0, NULL, 0);
        status = rt("UNKNOWN", frame, flen, &resp, &resp_len);
        check("unknown CMD rejected", status == RESP_ERR, NULL);
        free(resp);
    }
    printf("\n");

    /* ======================================================
     * Sustained throughput micro-benchmark — QUERY round-trips
     * ====================================================== */
    printf("[bench] QUERY round-trip throughput (full shape, L2)\n");
    double bench_dt = 0;
    int bench_n = 0;
    {
        unsigned long long s0 = g_bytes_sent, r0 = g_bytes_recv;
        int blen = 2 + vbytes;
        bodybuf[0] = METRIC_L2;
        bodybuf[1] = SHAPE_FULL;
        tstamp_t bt0; now(&bt0);
        for (int i = 0; i < BENCH_QUERIES; i++) {
            for (int d = 0; d < dim; d++) qvec[d] = randf();
            memcpy(bodybuf + 2, qvec, vbytes);
            int flen = build_frame(frame, CMD_QUERY, NULL, 0, bodybuf, blen);
            status = rt("QUERY-bench", frame, flen, &resp, &resp_len);
            free(resp);
        }
        bench_dt = ms_since(&bt0);
        bench_n = BENCH_QUERIES;
        unsigned long long sb = g_bytes_sent - s0;
        unsigned long long rb = g_bytes_recv - r0;
        double mb = (sb + rb) / (1024.0 * 1024.0);
        printf("    %d round-trips: total %.2f ms, avg %.3f ms, %.0f QPS\n",
               bench_n, bench_dt, bench_dt / bench_n, 1000.0 * bench_n / bench_dt);
        printf("    bytes sent=%.2f MB recv=%.2f MB total=%.2f MB throughput=%.1f MB/s\n",
               sb / (1024.0 * 1024.0), rb / (1024.0 * 1024.0), mb, mb * 1000.0 / bench_dt);
    }
    printf("\n");

    /* ======================================================
     * REPORT
     * ====================================================== */
    printf("====================================================================\n");
    printf("FINAL REPORT\n");
    printf("====================================================================\n");
    printf("  passed: %d\n", g_pass);
    printf("  failed: %d\n", g_fail);
    printf("\n");
    printf("  total frames sent: %llu (%.2f MB)\n", g_frames_sent, g_bytes_sent / (1024.0 * 1024.0));
    printf("  total frames recv: %llu (%.2f MB)\n", g_frames_recv, g_bytes_recv / (1024.0 * 1024.0));
    printf("\n");

    printf("  per-command latency:\n");
    printf("  %-22s %6s %10s %10s %10s %12s %12s\n",
           "command", "n", "min ms", "avg ms", "max ms", "sent KB", "recv KB");
    printf("  %-22s %6s %10s %10s %10s %12s %12s\n",
           "-------", "---", "------", "------", "------", "-------", "-------");
    for (int i = 0; i < g_stat_n; i++) {
        stat_t *s = &g_stats[i];
        if (s->n == 0) continue;
        printf("  %-22s %6d %10.4f %10.4f %10.4f %12.2f %12.2f\n",
               s->name, s->n, s->min_ms, s->total_ms / s->n, s->max_ms,
               s->bytes_sent / 1024.0, s->bytes_recv / 1024.0);
    }
    printf("====================================================================\n");

    /* useful one-liners for piping into other tools */
    if (bench_n > 0) {
        printf("\n  metrics: query_avg_ms=%.3f query_qps=%.0f labeled_push_per_s=%.0f tests_passed=%d tests_failed=%d\n",
               bench_dt / bench_n, 1000.0 * bench_n / bench_dt,
               batch_ok > 0 ? batch_ok / (g_stats[0].total_ms / 1000.0 + 0.001) : 0.0,
               g_pass, g_fail);
    }

    free(frame);
    free(bodybuf);
    free(vec);
    free(qvec);
    free(pushed_idx);
    free(data_blob);
    CloseHandle(g_pipe);
    return g_fail == 0 ? 0 : 1;
}
