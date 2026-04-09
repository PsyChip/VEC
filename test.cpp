/*
 * Curated by @PsyChip
 * root@psychip.net
 * April 2026
 *
 * test.cpp - test client for vec
 *
 * Usage: test <name> <dim> [port]
 *
 * Connects to vec via TCP, pushes 50 random vectors,
 * then queries with both L2 (pull) and cosine (cpull) to show the difference.
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

static void print_results(const char *resp, int pick) {
    char buf[4096];
    strncpy(buf, resp, sizeof(buf) - 1);
    buf[sizeof(buf) - 1] = '\0';

    printf("  rank  index     distance\n");
    printf("  ----  --------  ----------\n");

    int rank = 1;
    char *tok = strtok(buf, ",\n");
    while (tok) {
        int idx;
        float dist;
        if (sscanf(tok, "%d:%f", &idx, &dist) == 2) {
            printf("  %4d  %8d  %.6f%s\n", rank, idx, dist, idx == pick ? "  <-- MATCH" : "");
            rank++;
        }
        tok = strtok(NULL, ",\n");
    }
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: test <name> <dim> [port]\n");
        return 1;
    }

    const char *name = argv[1];
    int dim = atoi(argv[2]);
    int port = DEFAULT_PORT;
    if (argc >= 4) port = atoi(argv[3]);

    if (dim <= 0) {
        fprintf(stderr, "ERROR: invalid dim\n");
        return 1;
    }

    printf("name=%s dim=%d port=%d vectors=%d\n", name, dim, port, NUM_VECTORS);
    srand((unsigned)time(NULL));

    size_t total_floats = (size_t)NUM_VECTORS * dim;
    float *vectors = (float *)malloc(total_floats * sizeof(float));
    for (size_t i = 0; i < total_floats; i++) vectors[i] = randf();

    int pick = rand() % NUM_VECTORS;
    float *query = (float *)malloc(dim * sizeof(float));
    memcpy(query, vectors + (size_t)pick * dim, dim * sizeof(float));

    for (int d = 0; d < dim; d++) query[d] += randf() * 0.0001f;

    printf("query is noisy copy of vector #%d\n\n", pick);

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
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = htons((unsigned short)port);

    if (connect(s, (struct sockaddr *)&addr, sizeof(addr)) == SOCKET_ERROR) {
        fprintf(stderr, "ERROR: connect() failed - is vec running on port %d?\n", port);
        closesocket(s);
        WSACleanup();
        return 1;
    }
    printf("connected to 127.0.0.1:%d\n", port);

    char *cmd = (char *)malloc(BUF_SIZE);
    char resp[4096];

    printf("pushing %d vectors...\n", NUM_VECTORS);
    LARGE_INTEGER freq, t0, t1;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&t0);

    for (int i = 0; i < NUM_VECTORS; i++) {
        char *p = cmd;
        p += sprintf(p, "push ");
        for (int d = 0; d < dim; d++) {
            if (d > 0) *p++ = ',';
            p += sprintf(p, "%.6f", vectors[i * dim + d]);
        }
        *p++ = '\n';
        *p = '\0';
        sock_command(s, cmd, resp, sizeof(resp));
        if (strncmp(resp, "err", 3) == 0) {
            fprintf(stderr, "ERROR on push %d: %s", i, resp);
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
    print_results(resp, pick);
    printf("  time: %.3f ms\n\n", (double)(t1.QuadPart - t0.QuadPart) * 1000.0 / freq.QuadPart);

    build_query_cmd(cmd, "cpull", query, dim);
    printf("[COSINE] cpull:\n");
    QueryPerformanceCounter(&t0);
    sock_command(s, cmd, resp, sizeof(resp));
    QueryPerformanceCounter(&t1);
    print_results(resp, pick);
    printf("  time: %.3f ms\n\n", (double)(t1.QuadPart - t0.QuadPart) * 1000.0 / freq.QuadPart);

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
