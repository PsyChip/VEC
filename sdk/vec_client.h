/*
 * VEC 2.0 C++ Client SDK (binary frame protocol)
 *
 * VEC 2.0 is a clean break from 1.x. SDKs from 1.x are not wire-compatible.
 * See PROTOCOL-2.0.md for the full spec.
 *
 * Usage:
 *   VecClient vec;
 *   vec.connect_tcp("localhost", 1920);
 *
 *   // Push: vector required; data requires a label
 *   int idx = vec.push(vec_f32, dim);
 *   int idx = vec.push("img/cat.jpg", vec_f32, dim);
 *   int idx = vec.push("img/cat.jpg", vec_f32, dim, jpeg_bytes, jpeg_len);
 *
 *   // Query — returns N records into a caller-provided VecRecord array.
 *   // The records own malloc'd buffers for label/data/vector — call vec_free_record on each.
 *   VecRecord rs[10];
 *   int n = vec.query(vec_f32, dim, rs, 10);
 *   int n = vec.query(vec_f32, dim, rs, 10, 1, VEC_SHAPE_LABEL); // cosine, label-only
 *
 *   int n = vec.qid(42, rs, 10);
 *   int n = vec.qid("img/cat.jpg", rs, 10, true);
 *
 *   int n = vec.get(42, rs, 10);
 *   int n = vec.get_batch(idx_array, idx_count, rs, 10);
 *   int n = vec.get("img/cat.jpg", rs, 10);
 *
 *   for (int i = 0; i < n; i++) vec_free_record(&rs[i]);
 *
 *   vec.set_data(42, jpeg_bytes, jpeg_len);
 *   unsigned char *blob; unsigned int blob_len;
 *   vec.get_data(42, &blob, &blob_len);
 *   free(blob);
 *
 *   vec.update(42, vec_f32, dim);
 *   vec.set_label(42, "img/cat.jpg");
 *   vec.delete_index(42);                    // also clears label + data
 *   vec.undo();                               // also clears last label + data
 *
 *   unsigned int saved, crc;
 *   vec.save(&saved, &crc);
 *   VecInfo i; vec.info(&i);                  // i.protocol_version == 2
 *
 *   vec.close();
 *
 * Router mode:
 *   vec.setNamespace("tools");
 *
 * Build: link with ws2_32.lib (Windows) or nothing extra (Linux)
 */
#ifndef VEC_CLIENT_H
#define VEC_CLIENT_H

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
typedef SOCKET vec_sock_t;
#define VEC_INVALID_SOCK INVALID_SOCKET
#define vec_close_sock closesocket
#else
#include <sys/socket.h>
#include <sys/un.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <unistd.h>
typedef int vec_sock_t;
#define VEC_INVALID_SOCK -1
#define vec_close_sock ::close
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* protocol */
#define VEC_BIN_MAGIC        0xF0
#define VEC_PROTOCOL_VERSION 0x02

/* commands */
#define VEC_CMD_PUSH      0x01
#define VEC_CMD_QUERY     0x02
#define VEC_CMD_GET       0x04
#define VEC_CMD_UPDATE    0x06
#define VEC_CMD_DELETE    0x07
#define VEC_CMD_LABEL     0x08
#define VEC_CMD_UNDO      0x09
#define VEC_CMD_SAVE      0x0A
#define VEC_CMD_CLUSTER   0x0D
#define VEC_CMD_DISTINCT  0x0E
#define VEC_CMD_REPRESENT 0x0F
#define VEC_CMD_INFO      0x10
#define VEC_CMD_QID       0x11
#define VEC_CMD_SET_DATA  0x13
#define VEC_CMD_GET_DATA  0x14

/* response status */
#define VEC_RESP_OK  0x00
#define VEC_RESP_ERR 0x01

/* shape mask bits */
#define VEC_SHAPE_VECTOR 0x01
#define VEC_SHAPE_LABEL  0x02
#define VEC_SHAPE_DATA   0x04
#define VEC_SHAPE_FULL   (VEC_SHAPE_VECTOR | VEC_SHAPE_LABEL | VEC_SHAPE_DATA)

/* GET mode */
#define VEC_GET_MODE_SINGLE 0x00
#define VEC_GET_MODE_BATCH  0x01

/* metric */
#define VEC_METRIC_L2     0x00
#define VEC_METRIC_COSINE 0x01

/* limits */
#define VEC_MAX_LABEL_BYTES 2048
#define VEC_MAX_DATA_BYTES  102400

/* one record from query / qid / get. label, data, vector are malloc'd if non-NULL. */
struct VecRecord {
    int index;
    float distance;          /* valid only for query/qid; 0.0f otherwise */
    char *label;             /* NULL if shape didn't include or no label */
    unsigned int label_len;
    unsigned char *data;     /* NULL if shape didn't include or no data */
    unsigned int data_len;
    float *vector;           /* NULL if shape didn't include vector */
    int dim;                 /* number of f32 in vector, 0 if NULL */
};

inline void vec_free_record(VecRecord *r) {
    if (!r) return;
    free(r->label);  r->label  = NULL; r->label_len = 0;
    free(r->data);   r->data   = NULL; r->data_len  = 0;
    free(r->vector); r->vector = NULL; r->dim       = 0;
}

struct VecInfo {
    int dim;
    int count;
    int deleted;
    int fmt;                 /* 0=f32, 1=f16 */
    long long mtime;         /* unix epoch seconds */
    unsigned int crc;
    int crc_ok;              /* 0=mismatch, 1=ok, 2=unknown */
    char name[256];
    int protocol_version;    /* always 2 in this client */
};

class VecClient {
    vec_sock_t sock;
    char ns_buf[128];
    int ns_len;
    int dim_cache;

    int recv_exact(void *buf, int len) {
        int total = 0;
        while (total < len) {
            int r = recv(sock, (char *)buf + total, len - total, 0);
            if (r <= 0) return -1;
            total += r;
        }
        return total;
    }

    int send_all(const char *data, int len) {
        int sent = 0;
        while (sent < len) {
            int r = send(sock, data + sent, len - sent, 0);
            if (r <= 0) return -1;
            sent += r;
        }
        return sent;
    }

    /* build and send a 2.0 binary frame: F0 <2B ns_len> [ns] <CMD> <2B label_len> [label] <4B body_len> [body] */
    int send_frame(unsigned char cmd, const char *label, int label_len,
                   const char *body, int body_len) {
        int frame_len = 1 + 2 + ns_len + 1 + 2 + label_len + 4 + body_len;
        char *frame = (char *)malloc(frame_len);
        if (!frame) return -1;
        char *p = frame;
        *p++ = (char)VEC_BIN_MAGIC;
        unsigned short ns = (unsigned short)ns_len;
        memcpy(p, &ns, 2); p += 2;
        if (ns_len > 0) { memcpy(p, ns_buf, ns_len); p += ns_len; }
        *p++ = (char)cmd;
        unsigned short lbl = (unsigned short)label_len;
        memcpy(p, &lbl, 2); p += 2;
        if (label_len > 0) { memcpy(p, label, label_len); p += label_len; }
        unsigned int blen = (unsigned int)body_len;
        memcpy(p, &blen, 4); p += 4;
        if (body_len > 0) { memcpy(p, body, body_len); p += body_len; }
        int rc = send_all(frame, frame_len);
        free(frame);
        return rc;
    }

    /* receive a 2.0 response envelope into a malloc'd body buffer.
       returns body length, or -1 on error. on RESP_ERR returns -2 and copies
       the error message into err_buf (NUL-terminated, truncated to err_size). */
    int recv_response(unsigned char **out_body, char *err_buf, int err_size) {
        unsigned char hdr[5];
        if (recv_exact(hdr, 5) < 0) return -1;
        unsigned char status = hdr[0];
        unsigned int body_len;
        memcpy(&body_len, hdr + 1, 4);
        unsigned char *body = NULL;
        if (body_len > 0) {
            body = (unsigned char *)malloc(body_len);
            if (!body) return -1;
            if (recv_exact(body, body_len) < 0) { free(body); return -1; }
        }
        if (status == VEC_RESP_ERR) {
            if (err_buf && err_size > 0) {
                int n = (int)body_len < err_size - 1 ? (int)body_len : err_size - 1;
                if (n > 0) memcpy(err_buf, body, n);
                err_buf[n > 0 ? n : 0] = '\0';
            }
            free(body);
            return -2;
        }
        *out_body = body;
        return (int)body_len;
    }

    /* parse a result body produced by build_result_body on the server side. */
    int parse_records(const unsigned char *body, int body_len, unsigned char shape, int dim,
                      int with_distance, VecRecord *out, int max_records) {
        if (body_len < 4) return 0;
        unsigned int count;
        memcpy(&count, body, 4);
        int off = 4;
        int n = (int)count < max_records ? (int)count : max_records;
        for (int i = 0; i < n; i++) {
            VecRecord *r = &out[i];
            memset(r, 0, sizeof(*r));
            memcpy(&r->index, body + off, 4); off += 4;
            if (with_distance) { memcpy(&r->distance, body + off, 4); off += 4; }
            if (shape & VEC_SHAPE_LABEL) {
                memcpy(&r->label_len, body + off, 4); off += 4;
                if (r->label_len > 0) {
                    r->label = (char *)malloc(r->label_len + 1);
                    if (r->label) {
                        memcpy(r->label, body + off, r->label_len);
                        r->label[r->label_len] = '\0';
                    }
                    off += r->label_len;
                }
            }
            if (shape & VEC_SHAPE_DATA) {
                memcpy(&r->data_len, body + off, 4); off += 4;
                if (r->data_len > 0) {
                    r->data = (unsigned char *)malloc(r->data_len);
                    if (r->data) memcpy(r->data, body + off, r->data_len);
                    off += r->data_len;
                }
            }
            if (shape & VEC_SHAPE_VECTOR) {
                r->dim = dim;
                r->vector = (float *)malloc((size_t)dim * sizeof(float));
                if (r->vector) memcpy(r->vector, body + off, (size_t)dim * sizeof(float));
                off += dim * 4;
            }
        }
        return n;
    }

    int ensure_dim() {
        if (dim_cache > 0) return dim_cache;
        VecInfo i;
        if (info(&i) < 0) return -1;
        dim_cache = i.dim;
        return dim_cache;
    }

public:
    VecClient() : sock(VEC_INVALID_SOCK), ns_len(0), dim_cache(0) { ns_buf[0] = '\0'; }

    void setNamespace(const char *ns) {
        if (ns && ns[0]) {
            ns_len = (int)strlen(ns);
            if (ns_len > (int)sizeof(ns_buf) - 1) ns_len = (int)sizeof(ns_buf) - 1;
            memcpy(ns_buf, ns, ns_len);
            ns_buf[ns_len] = '\0';
        } else {
            ns_len = 0;
            ns_buf[0] = '\0';
        }
    }

    int connect_tcp(const char *host, int port) {
#ifdef _WIN32
        WSADATA wsa;
        WSAStartup(MAKEWORD(2, 2), &wsa);
#endif
        sock = socket(AF_INET, SOCK_STREAM, 0);
        if (sock == VEC_INVALID_SOCK) return -1;

        struct sockaddr_in addr;
        memset(&addr, 0, sizeof(addr));
        addr.sin_family = AF_INET;
        addr.sin_port = htons((unsigned short)port);

        if (inet_pton(AF_INET, host, &addr.sin_addr) != 1) {
            struct hostent *he = gethostbyname(host);
            if (!he) { vec_close_sock(sock); sock = VEC_INVALID_SOCK; return -1; }
            memcpy(&addr.sin_addr, he->h_addr_list[0], he->h_length);
        }

        if (::connect(sock, (struct sockaddr *)&addr, sizeof(addr)) != 0) {
            vec_close_sock(sock); sock = VEC_INVALID_SOCK; return -1;
        }
        return 0;
    }

#ifndef _WIN32
    int connect_unix(const char *name) {
        char path[256];
        snprintf(path, sizeof(path), "/tmp/vec_%s.sock", name);
        sock = socket(AF_UNIX, SOCK_STREAM, 0);
        if (sock < 0) return -1;
        struct sockaddr_un addr;
        memset(&addr, 0, sizeof(addr));
        addr.sun_family = AF_UNIX;
        strncpy(addr.sun_path, path, sizeof(addr.sun_path) - 1);
        if (::connect(sock, (struct sockaddr *)&addr, sizeof(addr)) != 0) {
            ::close(sock); sock = VEC_INVALID_SOCK; return -1;
        }
        return 0;
    }
#endif

    /* PUSH variants */

    int push(const float *vec, int dim) {
        return push(NULL, 0, vec, dim, NULL, 0);
    }
    int push(const char *label, const float *vec, int dim) {
        return push(label, label ? (int)strlen(label) : 0, vec, dim, NULL, 0);
    }
    int push(const char *label, const float *vec, int dim,
             const unsigned char *data, unsigned int data_len) {
        return push(label, label ? (int)strlen(label) : 0, vec, dim, data, data_len);
    }
    int push(const char *label, int label_len, const float *vec, int dim,
             const unsigned char *data, unsigned int data_len) {
        if (data_len > 0 && label_len <= 0) return -1;
        if (data_len > VEC_MAX_DATA_BYTES) return -1;
        int vbytes = dim * (int)sizeof(float);
        int blen = vbytes + 4 + (int)data_len;
        char *body = (char *)malloc(blen);
        if (!body) return -1;
        memcpy(body, vec, vbytes);
        memcpy(body + vbytes, &data_len, 4);
        if (data_len > 0) memcpy(body + vbytes + 4, data, data_len);
        send_frame(VEC_CMD_PUSH, label, label_len, body, blen);
        free(body);
        unsigned char *resp = NULL;
        char err[128];
        int n = recv_response(&resp, err, sizeof(err));
        if (n < 0) return -1;
        int slot = -1;
        if (n >= 4) memcpy(&slot, resp, 4);
        free(resp);
        return slot;
    }

    /* QUERY: nearest-neighbor by query vector */

    int query(const float *vec, int dim, VecRecord *out, int max_records) {
        return query(vec, dim, out, max_records, /*cosine*/0, VEC_SHAPE_FULL);
    }
    int query(const float *vec, int dim, VecRecord *out, int max_records,
              int cosine, unsigned char shape) {
        char *body = (char *)malloc(2 + dim * sizeof(float));
        if (!body) return -1;
        body[0] = (char)(cosine ? VEC_METRIC_COSINE : VEC_METRIC_L2);
        body[1] = (char)shape;
        memcpy(body + 2, vec, dim * sizeof(float));
        send_frame(VEC_CMD_QUERY, NULL, 0, body, 2 + dim * (int)sizeof(float));
        free(body);
        unsigned char *resp = NULL;
        char err[128];
        int n = recv_response(&resp, err, sizeof(err));
        if (n < 0) return -1;
        int rc = parse_records(resp, n, shape, dim, /*with_distance*/1, out, max_records);
        free(resp);
        return rc;
    }

    /* QID: nearest-neighbor by stored vector */

    int qid(int index, VecRecord *out, int max_records) {
        return qid_impl(index, NULL, 0, out, max_records, 0, VEC_SHAPE_FULL);
    }
    int qid(int index, VecRecord *out, int max_records, int cosine, unsigned char shape) {
        return qid_impl(index, NULL, 0, out, max_records, cosine, shape);
    }
    int qid(const char *label, VecRecord *out, int max_records) {
        return qid_impl(-1, label, label ? (int)strlen(label) : 0, out, max_records, 0, VEC_SHAPE_FULL);
    }
    int qid(const char *label, VecRecord *out, int max_records, int cosine, unsigned char shape = VEC_SHAPE_FULL) {
        return qid_impl(-1, label, label ? (int)strlen(label) : 0, out, max_records, cosine, shape);
    }

    int qid_impl(int index, const char *label, int label_len,
                 VecRecord *out, int max_records, int cosine, unsigned char shape) {
        int dim = ensure_dim();
        if (dim < 0) return -1;
        char head[2] = { (char)(cosine ? VEC_METRIC_COSINE : VEC_METRIC_L2), (char)shape };
        if (label_len > 0) {
            send_frame(VEC_CMD_QID, label, label_len, head, 2);
        } else {
            char body[6];
            memcpy(body, head, 2);
            memcpy(body + 2, &index, 4);
            send_frame(VEC_CMD_QID, NULL, 0, body, 6);
        }
        unsigned char *resp = NULL;
        char err[128];
        int n = recv_response(&resp, err, sizeof(err));
        if (n < 0) return -1;
        int rc = parse_records(resp, n, shape, dim, 1, out, max_records);
        free(resp);
        return rc;
    }

    /* GET single by index */
    int get(int index, VecRecord *out, int max_records) {
        return get(index, out, max_records, VEC_SHAPE_FULL);
    }
    int get(int index, VecRecord *out, int max_records, unsigned char shape) {
        int dim = ensure_dim();
        if (dim < 0) return -1;
        char body[6];
        body[0] = (char)VEC_GET_MODE_SINGLE;
        body[1] = (char)shape;
        memcpy(body + 2, &index, 4);
        send_frame(VEC_CMD_GET, NULL, 0, body, 6);
        unsigned char *resp = NULL;
        char err[128];
        int n = recv_response(&resp, err, sizeof(err));
        if (n < 0) return -1;
        int rc = parse_records(resp, n, shape, dim, 0, out, max_records);
        free(resp);
        return rc;
    }

    /* GET single by label */
    int get(const char *label, VecRecord *out, int max_records) {
        return get(label, out, max_records, VEC_SHAPE_FULL);
    }
    int get(const char *label, VecRecord *out, int max_records, unsigned char shape) {
        int dim = ensure_dim();
        if (dim < 0) return -1;
        char head[2] = { (char)VEC_GET_MODE_SINGLE, (char)shape };
        send_frame(VEC_CMD_GET, label, (int)strlen(label), head, 2);
        unsigned char *resp = NULL;
        char err[128];
        int n = recv_response(&resp, err, sizeof(err));
        if (n < 0) return -1;
        int rc = parse_records(resp, n, shape, dim, 0, out, max_records);
        free(resp);
        return rc;
    }

    /* GET batch by indices */
    int get_batch(const int *indices, int count, VecRecord *out, int max_records) {
        return get_batch(indices, count, out, max_records, VEC_SHAPE_FULL);
    }
    int get_batch(const int *indices, int count, VecRecord *out, int max_records, unsigned char shape) {
        int dim = ensure_dim();
        if (dim < 0) return -1;
        int blen = 2 + 4 + count * 4;
        char *body = (char *)malloc(blen);
        if (!body) return -1;
        body[0] = (char)VEC_GET_MODE_BATCH;
        body[1] = (char)shape;
        unsigned int u = (unsigned int)count;
        memcpy(body + 2, &u, 4);
        memcpy(body + 6, indices, count * 4);
        send_frame(VEC_CMD_GET, NULL, 0, body, blen);
        free(body);
        unsigned char *resp = NULL;
        char err[128];
        int n = recv_response(&resp, err, sizeof(err));
        if (n < 0) return -1;
        int rc = parse_records(resp, n, shape, dim, 0, out, max_records);
        free(resp);
        return rc;
    }

    int update(int index, const float *vec, int dim) {
        int blen = 4 + dim * (int)sizeof(float);
        char *body = (char *)malloc(blen);
        if (!body) return -1;
        memcpy(body, &index, 4);
        memcpy(body + 4, vec, dim * sizeof(float));
        send_frame(VEC_CMD_UPDATE, NULL, 0, body, blen);
        free(body);
        unsigned char *resp = NULL;
        char err[128];
        int n = recv_response(&resp, err, sizeof(err));
        free(resp);
        return n < 0 ? -1 : 0;
    }
    int update(const char *label, const float *vec, int dim) {
        send_frame(VEC_CMD_UPDATE, label, (int)strlen(label),
                   (const char *)vec, dim * (int)sizeof(float));
        unsigned char *resp = NULL;
        char err[128];
        int n = recv_response(&resp, err, sizeof(err));
        free(resp);
        return n < 0 ? -1 : 0;
    }

    int set_label(int index, const char *label) {
        int label_len = label ? (int)strlen(label) : 0;
        send_frame(VEC_CMD_LABEL, label, label_len, (const char *)&index, 4);
        unsigned char *resp = NULL;
        char err[128];
        int n = recv_response(&resp, err, sizeof(err));
        free(resp);
        return n < 0 ? -1 : 0;
    }

    int delete_index(int index) {
        send_frame(VEC_CMD_DELETE, NULL, 0, (const char *)&index, 4);
        unsigned char *resp = NULL;
        char err[128];
        int n = recv_response(&resp, err, sizeof(err));
        free(resp);
        return n < 0 ? -1 : 0;
    }
    int delete_label(const char *label) {
        send_frame(VEC_CMD_DELETE, label, (int)strlen(label), NULL, 0);
        unsigned char *resp = NULL;
        char err[128];
        int n = recv_response(&resp, err, sizeof(err));
        free(resp);
        return n < 0 ? -1 : 0;
    }

    int undo() {
        send_frame(VEC_CMD_UNDO, NULL, 0, NULL, 0);
        unsigned char *resp = NULL;
        char err[128];
        int n = recv_response(&resp, err, sizeof(err));
        free(resp);
        return n < 0 ? -1 : 0;
    }

    int save(unsigned int *out_count, unsigned int *out_crc) {
        send_frame(VEC_CMD_SAVE, NULL, 0, NULL, 0);
        unsigned char *resp = NULL;
        char err[128];
        int n = recv_response(&resp, err, sizeof(err));
        if (n < 0) { free(resp); return -1; }
        if (n >= 8) {
            if (out_count) memcpy(out_count, resp, 4);
            if (out_crc)   memcpy(out_crc,   resp + 4, 4);
        }
        free(resp);
        return 0;
    }

    int info(VecInfo *out) {
        send_frame(VEC_CMD_INFO, NULL, 0, NULL, 0);
        unsigned char *resp = NULL;
        char err[128];
        int n = recv_response(&resp, err, sizeof(err));
        if (n < 0) return -1;
        int off = 0;
        memcpy(&out->dim, resp + off, 4); off += 4;
        memcpy(&out->count, resp + off, 4); off += 4;
        memcpy(&out->deleted, resp + off, 4); off += 4;
        out->fmt = (unsigned char)resp[off]; off += 1;
        memcpy(&out->mtime, resp + off, 8); off += 8;
        memcpy(&out->crc, resp + off, 4); off += 4;
        out->crc_ok = (unsigned char)resp[off]; off += 1;
        unsigned int name_len;
        memcpy(&name_len, resp + off, 4); off += 4;
        if (name_len > sizeof(out->name) - 1) name_len = sizeof(out->name) - 1;
        if (name_len > 0) memcpy(out->name, resp + off, name_len);
        out->name[name_len] = '\0';
        off += name_len;
        out->protocol_version = (unsigned char)resp[off];
        free(resp);
        return 0;
    }

    int set_data(int index, const unsigned char *data, unsigned int data_len) {
        if (data_len > VEC_MAX_DATA_BYTES) return -1;
        int blen = 8 + (int)data_len;
        char *body = (char *)malloc(blen);
        if (!body) return -1;
        memcpy(body, &index, 4);
        memcpy(body + 4, &data_len, 4);
        if (data_len > 0) memcpy(body + 8, data, data_len);
        send_frame(VEC_CMD_SET_DATA, NULL, 0, body, blen);
        free(body);
        unsigned char *resp = NULL;
        char err[128];
        int n = recv_response(&resp, err, sizeof(err));
        free(resp);
        return n < 0 ? -1 : 0;
    }
    int set_data(const char *label, const unsigned char *data, unsigned int data_len) {
        if (data_len > VEC_MAX_DATA_BYTES) return -1;
        int blen = 4 + (int)data_len;
        char *body = (char *)malloc(blen);
        if (!body) return -1;
        memcpy(body, &data_len, 4);
        if (data_len > 0) memcpy(body + 4, data, data_len);
        send_frame(VEC_CMD_SET_DATA, label, (int)strlen(label), body, blen);
        free(body);
        unsigned char *resp = NULL;
        char err[128];
        int n = recv_response(&resp, err, sizeof(err));
        free(resp);
        return n < 0 ? -1 : 0;
    }

    /* get_data: caller frees *out_data with free() */
    int get_data(int index, unsigned char **out_data, unsigned int *out_len) {
        send_frame(VEC_CMD_GET_DATA, NULL, 0, (const char *)&index, 4);
        return recv_data_response(out_data, out_len);
    }
    int get_data(const char *label, unsigned char **out_data, unsigned int *out_len) {
        send_frame(VEC_CMD_GET_DATA, label, (int)strlen(label), NULL, 0);
        return recv_data_response(out_data, out_len);
    }

    int recv_data_response(unsigned char **out_data, unsigned int *out_len) {
        unsigned char *resp = NULL;
        char err[128];
        int n = recv_response(&resp, err, sizeof(err));
        if (n < 0) { *out_data = NULL; *out_len = 0; return -1; }
        unsigned int dlen = 0;
        if (n >= 4) memcpy(&dlen, resp, 4);
        *out_len = dlen;
        if (dlen > 0) {
            *out_data = (unsigned char *)malloc(dlen);
            if (*out_data) memcpy(*out_data, resp + 4, dlen);
        } else {
            *out_data = NULL;
        }
        free(resp);
        return 0;
    }

    /* cluster/distinct/represent — body remains legacy text inside binary envelope.
       caller gets a text dump in out_buf (NUL-terminated, truncated). */
    int cluster_raw(float eps, int cosine, int min_pts, char *out_buf, int out_size) {
        char body[9];
        memcpy(body, &eps, 4);
        body[4] = (char)(cosine ? VEC_METRIC_COSINE : VEC_METRIC_L2);
        memcpy(body + 5, &min_pts, 4);
        send_frame(VEC_CMD_CLUSTER, NULL, 0, body, 9);
        return copy_text_body(out_buf, out_size);
    }
    int distinct_raw(int k, int cosine, char *out_buf, int out_size) {
        char body[5];
        memcpy(body, &k, 4);
        body[4] = (char)(cosine ? VEC_METRIC_COSINE : VEC_METRIC_L2);
        send_frame(VEC_CMD_DISTINCT, NULL, 0, body, 5);
        return copy_text_body(out_buf, out_size);
    }
    int represent_raw(float eps, int cosine, int min_pts, char *out_buf, int out_size) {
        char body[9];
        memcpy(body, &eps, 4);
        body[4] = (char)(cosine ? VEC_METRIC_COSINE : VEC_METRIC_L2);
        memcpy(body + 5, &min_pts, 4);
        send_frame(VEC_CMD_REPRESENT, NULL, 0, body, 9);
        return copy_text_body(out_buf, out_size);
    }

    int copy_text_body(char *out_buf, int out_size) {
        unsigned char *resp = NULL;
        char err[256];
        int n = recv_response(&resp, err, sizeof(err));
        if (n < 0) {
            if (out_buf && out_size > 0) {
                int copy = (int)strlen(err);
                if (copy >= out_size) copy = out_size - 1;
                memcpy(out_buf, err, copy);
                out_buf[copy] = '\0';
            }
            free(resp);
            return -1;
        }
        if (out_buf && out_size > 0) {
            int copy = n < out_size - 1 ? n : out_size - 1;
            if (copy > 0) memcpy(out_buf, resp, copy);
            out_buf[copy > 0 ? copy : 0] = '\0';
        }
        free(resp);
        return n;
    }

    void close() {
        if (sock != VEC_INVALID_SOCK) {
            vec_close_sock(sock);
            sock = VEC_INVALID_SOCK;
        }
    }
};

#endif /* VEC_CLIENT_H */
