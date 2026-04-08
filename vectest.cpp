/*
 * vectest.cpp — test client for vec
 *
 * Usage: vectest <name> <dim>
 *
 * Connects to vec via TCP port 1920, pushes 50 random vectors,
 * then queries a noisy copy of one and checks if it comes back #1.
 *
 * Build:
 *   cl /O2 /EHsc vectest.cpp /Fe:vectest.exe ws2_32.lib
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

#define TCP_PORT    1920
#define NUM_VECTORS 50
#define BUF_SIZE    (1 << 20)

static SOCKET sock_connect()
{
    WSADATA wsa;
    WSAStartup(MAKEWORD(2, 2), &wsa);

    SOCKET s = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (s == INVALID_SOCKET) {
        fprintf(stderr, "ERROR: socket() failed\n");
        exit(1);
    }

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = htons(TCP_PORT);

    if (connect(s, (struct sockaddr*)&addr, sizeof(addr)) == SOCKET_ERROR) {
        fprintf(stderr, "ERROR: connect() failed — is vec running?\n");
        closesocket(s);
        WSACleanup();
        exit(1);
    }
    return s;
}

/* Send a line, read response lines until "OK\n" or "ERR" or "BYE" */
static int sock_command(SOCKET s, const char* cmd, char* resp, int resp_size)
{
    int len = (int)strlen(cmd);
    send(s, cmd, len, 0);

    int total = 0;
    while (total < resp_size - 1) {
        int r = recv(s, resp + total, resp_size - total - 1, 0);
        if (r <= 0) break;
        total += r;
        resp[total] = '\0';
        if (strstr(resp, "OK\n") || strstr(resp, "ERR") || strstr(resp, "BYE"))
            break;
    }
    return total;
}

static float randf()
{
    return ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
}

int main(int argc, char** argv)
{
    if (argc < 3) {
        fprintf(stderr, "Usage: vectest <name> <dim>\n");
        return 1;
    }

    const char* name = argv[1];
    int dim = atoi(argv[2]);
    if (dim <= 0) {
        fprintf(stderr, "ERROR: invalid dim\n");
        return 1;
    }

    printf("[vectest] name=%s dim=%d vectors=%d\n", name, dim, NUM_VECTORS);
    srand((unsigned)time(NULL));

    /* generate random vectors */
    size_t total_floats = (size_t)NUM_VECTORS * dim;
    float* vectors = (float*)malloc(total_floats * sizeof(float));
    for (size_t i = 0; i < total_floats; i++)
        vectors[i] = randf();

    /* pick one to query later */
    int pick = rand() % NUM_VECTORS;
    float* query = (float*)malloc(dim * sizeof(float));
    memcpy(query, vectors + (size_t)pick * dim, dim * sizeof(float));

    /* add tiny noise */
    for (int d = 0; d < dim; d++)
        query[d] += randf() * 0.0001f;

    printf("[vectest] will query noisy copy of vector #%d\n", pick);

    /* connect */
    SOCKET s = sock_connect();
    printf("[vectest] connected to 127.0.0.1:%d\n", TCP_PORT);

    char* cmd = (char*)malloc(BUF_SIZE);
    char* resp = (char*)malloc(BUF_SIZE);

    /* push all vectors */
    printf("[vectest] pushing %d vectors...\n", NUM_VECTORS);
    LARGE_INTEGER freq, t0, t1;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&t0);

    for (int i = 0; i < NUM_VECTORS; i++) {
        char* p = cmd;
        p += sprintf(p, "push ");
        for (int d = 0; d < dim; d++) {
            if (d > 0) *p++ = ',';
            p += sprintf(p, "%.6f", vectors[i * dim + d]);
        }
        *p++ = '\n';
        *p = '\0';

        sock_command(s, cmd, resp, BUF_SIZE);

        if (strstr(resp, "ERR")) {
            fprintf(stderr, "ERROR on push %d: %s", i, resp);
            goto done;
        }
    }

    QueryPerformanceCounter(&t1);
    printf("[vectest] pushed %d vectors in %.1f ms\n", NUM_VECTORS,
           (double)(t1.QuadPart - t0.QuadPart) * 1000.0 / freq.QuadPart);

    /* query */
    {
        char* p = cmd;
        p += sprintf(p, "pull ");
        for (int d = 0; d < dim; d++) {
            if (d > 0) *p++ = ',';
            p += sprintf(p, "%.6f", query[d]);
        }
        *p++ = '\n';
        *p = '\0';

        printf("[vectest] querying...\n");
        QueryPerformanceCounter(&t0);
        sock_command(s, cmd, resp, BUF_SIZE);
        QueryPerformanceCounter(&t1);

        printf("[vectest] query took %.3f ms\n",
               (double)(t1.QuadPart - t0.QuadPart) * 1000.0 / freq.QuadPart);

        printf("\n[vectest] results (expected #%d at rank 1):\n", pick);
        printf("  rank  index     distance\n");
        printf("  ----  --------  ----------\n");

        int rank = 1;
        char* line = strtok(resp, "\n");
        while (line) {
            if (strcmp(line, "OK") == 0) break;
            if (strncmp(line, "ERR", 3) == 0) {
                fprintf(stderr, "  ERROR: %s\n", line);
                break;
            }
            int idx;
            float dist;
            if (sscanf(line, "%d %f", &idx, &dist) == 2) {
                printf("  %4d  %8d  %.6f%s\n", rank, idx, dist,
                       idx == pick ? "  <-- MATCH" : "");
                rank++;
            }
            line = strtok(NULL, "\n");
        }
    }

    /* save */
    printf("\n[vectest] sending save...\n");
    sock_command(s, "save\n", resp, BUF_SIZE);
    printf("[vectest] %s", resp);

    /* count */
    sock_command(s, "count\n", resp, BUF_SIZE);
    printf("[vectest] count: %s", resp);

done:
    closesocket(s);
    WSACleanup();
    free(vectors);
    free(query);
    free(cmd);
    free(resp);

    printf("[vectest] done.\n");
    return 0;
}
