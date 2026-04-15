/*
 * Curated by @PsyChip
 * root@psychip.net
 * April 2026
 *
 * test.cpp - test client for vec
 *
 * Usage: test <name> <dim> [port]
 *
 * Connects to vec via TCP, pushes vectors with labels via bpush,
 * queries with text (pull/cpull) and binary (bpull/bcpull),
 * tests label override, verifies results.
 *
 * Build:
 *   cl /O2 /EHsc test.cpp /Fe:test.exe ws2_32.lib
 */

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define DEFAULT_PORT 1920
#define NUM_VECTORS  1000
#define BUF_SIZE     (1 << 20)

static int sock_readline(SOCKET s, char *buf, int buf_size) {
    int total = 0;
    while (total < buf_size - 1) {
        int r = recv(s, buf + total, 1, 0);
        if (r <= 0) break;
        total++;
        if (buf[total - 1] == '\n') break;
    }
    buf[total] = '\0';
    return total;
}

static int sock_command(SOCKET s, const char *cmd, char *resp, int resp_size) {
    int len = (int)strlen(cmd);
    send(s, cmd, len, 0);
    return sock_readline(s, resp, resp_size);
}

static float randf() {
    return ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
}

static void build_query_cmd(char *cmd, const char *prefix, const float *query, int dim) {
    char *p = cmd;
    p += sprintf(p, "%s ", prefix);
    for (int d = 0; d < dim; d++) {
        if (d > 0) *p++ = ',';
        p += sprintf(p, "%.6f", query[d]);
    }
    *p++ = '\n';
    *p = '\0';
}

static void print_results(const char *resp, const char *pick_label) {
    char buf[4096];
    strncpy(buf, resp, sizeof(buf) - 1);
    buf[sizeof(buf) - 1] = '\0';

    printf("  rank  label/index              distance\n");
    printf("  ----  -----------------------  ----------\n");

    int rank = 1;
    char *tok = strtok(buf, ",\n");
    while (tok) {
        char *last_colon = strrchr(tok, ':');
        if (!last_colon) { tok = strtok(NULL, ",\n"); continue; }
        *last_colon = '\0';
        const char *key = tok;
        float dist = (float)atof(last_colon + 1);
        int is_match = (pick_label && strcmp(key, pick_label) == 0);
        printf("  %4d  %-23s  %.6f%s\n", rank, key, dist, is_match ? "  <-- MATCH" : "");
        rank++;
        tok = strtok(NULL, ",\n");
    }
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: test <name> <dim> [host:port]\n");
        return 1;
    }

    const char *name = argv[1];
    int dim = atoi(argv[2]);
    char host[256] = "127.0.0.1";
    int port = DEFAULT_PORT;
    if (argc >= 4) {
        strncpy(host, argv[3], sizeof(host) - 1);
        char *colon = strchr(host, ':');
        if (colon) {
            *colon = '\0';
            port = atoi(colon + 1);
        }
    }

    if (dim <= 0) {
        fprintf(stderr, "ERROR: invalid dim\n");
        return 1;
    }

    printf("name=%s dim=%d host=%s port=%d vectors=%d\n", name, dim, host, port, NUM_VECTORS);
    srand((unsigned)time(NULL));

    size_t total_floats = (size_t)NUM_VECTORS * dim;
    float *vectors = (float *)malloc(total_floats * sizeof(float));
    for (size_t i = 0; i < total_floats; i++) vectors[i] = randf();

    int pick = rand() % NUM_VECTORS;
    float *query = (float *)malloc(dim * sizeof(float));
    memcpy(query, vectors + (size_t)pick * dim, dim * sizeof(float));

    for (int d = 0; d < dim; d++) query[d] += randf() * 0.0001f;

    /* generate labels for each vector */
    char pick_label[128];
    snprintf(pick_label, sizeof(pick_label), "test/vector_%d?dim=%d", pick, dim);
    printf("query is noisy copy of vector #%d (label=%s)\n\n", pick, pick_label);

    WSADATA wsa;
    WSAStartup(MAKEWORD(2, 2), &wsa);

    SOCKET s = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (s == INVALID_SOCKET) {
        fprintf(stderr, "ERROR: socket() failed\n");
        return 1;
    }

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    if (inet_pton(AF_INET, host, &addr.sin_addr) != 1) {
        struct hostent *he = gethostbyname(host);
        if (!he) {
            fprintf(stderr, "ERROR: cannot resolve '%s'\n", host);
            closesocket(s);
            WSACleanup();
            return 1;
        }
        memcpy(&addr.sin_addr, he->h_addr_list[0], he->h_length);
    }
    addr.sin_port = htons((unsigned short)port);

    if (connect(s, (struct sockaddr *)&addr, sizeof(addr)) == SOCKET_ERROR) {
        fprintf(stderr, "ERROR: connect() failed - is vec running on %s:%d?\n", host, port);
        closesocket(s);
        WSACleanup();
        return 1;
    }
    printf("connected to %s:%d\n", host, port);

    char *cmd = (char *)malloc(BUF_SIZE);
    char resp[4096];

    printf("pushing %d vectors via bpush...\n", NUM_VECTORS);
    LARGE_INTEGER freq, t0, t1;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&t0);

    for (int i = 0; i < NUM_VECTORS; i++) {
        /* send: bpush label\n<dim*4 bytes> */
        char header[256];
        int hlen = sprintf(header, "bpush test/vector_%d?dim=%d\n", i, dim);
        send(s, header, hlen, 0);

        const char *vec_data = (const char *)(vectors + i * dim);
        int vec_bytes = dim * (int)sizeof(float);
        int sent = 0;
        while (sent < vec_bytes) {
            int r = send(s, vec_data + sent, vec_bytes - sent, 0);
            if (r <= 0) { fprintf(stderr, "ERROR: send failed\n"); goto done; }
            sent += r;
        }

        sock_readline(s, resp, sizeof(resp));
        if (strncmp(resp, "err", 3) == 0) {
            fprintf(stderr, "ERROR on bpush %d: %s", i, resp);
            goto done;
        }
    }

    QueryPerformanceCounter(&t1);
    printf("pushed %d vectors in %.1f ms\n\n", NUM_VECTORS, (double)(t1.QuadPart - t0.QuadPart) * 1000.0 / freq.QuadPart);

    build_query_cmd(cmd, "pull", query, dim);
    printf("[L2] pull:\n");
    QueryPerformanceCounter(&t0);
    sock_command(s, cmd, resp, sizeof(resp));
    QueryPerformanceCounter(&t1);
    print_results(resp, pick_label);
    printf("  time: %.3f ms\n\n", (double)(t1.QuadPart - t0.QuadPart) * 1000.0 / freq.QuadPart);

    build_query_cmd(cmd, "cpull", query, dim);
    printf("[COSINE] cpull:\n");
    QueryPerformanceCounter(&t0);
    sock_command(s, cmd, resp, sizeof(resp));
    QueryPerformanceCounter(&t1);
    print_results(resp, pick_label);
    printf("  time: %.3f ms\n\n", (double)(t1.QuadPart - t0.QuadPart) * 1000.0 / freq.QuadPart);

    /* --- bpull: binary L2 query --- */
    {
        printf("[L2] bpull (binary):\n");
        send(s, "bpull\n", 6, 0);
        int vec_bytes = dim * (int)sizeof(float);
        int sent = 0;
        while (sent < vec_bytes) {
            int r = send(s, (const char *)query + sent, vec_bytes - sent, 0);
            if (r <= 0) { fprintf(stderr, "ERROR: send failed\n"); goto done; }
            sent += r;
        }
        QueryPerformanceCounter(&t0);
        sock_readline(s, resp, sizeof(resp));
        QueryPerformanceCounter(&t1);
        print_results(resp, pick_label);
        printf("  time: %.3f ms\n\n", (double)(t1.QuadPart - t0.QuadPart) * 1000.0 / freq.QuadPart);
    }

    /* --- bcpull: binary cosine query --- */
    {
        printf("[COSINE] bcpull (binary):\n");
        send(s, "bcpull\n", 7, 0);
        int vec_bytes = dim * (int)sizeof(float);
        int sent = 0;
        while (sent < vec_bytes) {
            int r = send(s, (const char *)query + sent, vec_bytes - sent, 0);
            if (r <= 0) { fprintf(stderr, "ERROR: send failed\n"); goto done; }
            sent += r;
        }
        QueryPerformanceCounter(&t0);
        sock_readline(s, resp, sizeof(resp));
        QueryPerformanceCounter(&t1);
        print_results(resp, pick_label);
        printf("  time: %.3f ms\n\n", (double)(t1.QuadPart - t0.QuadPart) * 1000.0 / freq.QuadPart);
    }

    /* --- label override test --- */
    {
        printf("label override test:\n");
        char lcmd[256];
        sprintf(lcmd, "label %d relabeled/vector_%d\n", pick, pick);
        sock_command(s, lcmd, resp, sizeof(resp));
        printf("  label %d -> %s", pick, resp);

        /* re-query to see updated label */
        build_query_cmd(cmd, "pull", query, dim);
        sock_command(s, cmd, resp, sizeof(resp));
        char relabel[128];
        snprintf(relabel, sizeof(relabel), "relabeled/vector_%d", pick);
        print_results(resp, relabel);
        printf("\n");
    }

    sock_command(s, "size\n", resp, sizeof(resp));
    printf("size: %s", resp);

done:
    closesocket(s);
    WSACleanup();
    free(vectors);
    free(query);
    free(cmd);
    printf("done.\n");
    return 0;
}
