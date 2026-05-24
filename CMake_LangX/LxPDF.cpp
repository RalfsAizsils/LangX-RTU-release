#include "LxPDF.h"
#include "miniz.h"
#include <fstream>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <iostream>

namespace {

std::string inflate_flate(const uint8_t* data, size_t len) {
    size_t out_cap = std::max(len * 4, (size_t)4096);
    for (int attempt = 0; attempt < 4; attempt++, out_cap *= 4) {
        std::vector<uint8_t> out(out_cap);
        mz_ulong actual = (mz_ulong)out_cap;
        int r = mz_uncompress(out.data(), &actual, data, (mz_ulong)len);
        if (r == MZ_OK) return std::string(out.begin(), out.begin() + actual);
        if (r != MZ_BUF_ERROR) break;
    }
    return {};
}

void collect_text(const std::string& cs, std::string& out) {
    auto skip_ws = [&](size_t p) -> size_t {
        while (p < cs.size() &&
               (cs[p]==' '||cs[p]=='\t'||cs[p]=='\r'||cs[p]=='\n')) p++;
        return p;
    };

    auto read_paren = [&](size_t& p) -> std::string {
        if (p >= cs.size() || cs[p] != '(') return {};
        p++;
        std::string s;
        int depth = 1;
        while (p < cs.size() && depth > 0) {
            unsigned char c = cs[p];
            if (c == '\\' && p + 1 < cs.size()) {
                p++; unsigned char nc = cs[p];
                if (nc == 'n') s += '\n';
                else if (nc == 'r') s += ' ';
                else if (nc == 't') s += '\t';
                else if (nc == '(' || nc == ')' || nc == '\\') s += nc;
                else if (nc >= '0' && nc <= '7') {
                    unsigned val = nc - '0';
                    for (int k = 0; k < 2 && p+1 < cs.size() && cs[p+1]>='0' && cs[p+1]<='7'; k++)
                        val = val * 8 + (cs[++p] - '0');
                    if (val >= 32 && val < 127) s += (char)val;
                }
            } else if (c == '(') { depth++; s += '('; }
              else if (c == ')') { if (--depth > 0) s += ')'; }
              else if (c >= 32 && c < 127) s += c;
            p++;
        }
        return s;
    };

    size_t i = 0;
    while (i < cs.size()) {
        i = skip_ws(i);
        if (i >= cs.size()) break;

        if (cs[i] == '(') {
            size_t start = i;
            std::string s = read_paren(i);
            i = skip_ws(i);
            if (i < cs.size()) {
                char op = cs[i];
                if (op == 'T' && i+1 < cs.size() && cs[i+1] == 'j') {
                    out += s + ' '; i += 2;
                } else if (op == '\'' || op == '"') {
                    out += '\n'; out += s + ' '; i++;
                } else {
                    i = start + 1;
                }
            }

        } else if (cs[i] == '[') {
            i++;
            std::string arr;
            while (i < cs.size() && cs[i] != ']') {
                i = skip_ws(i);
                if (i < cs.size() && cs[i] == '(') {
                    arr += read_paren(i);
                } else if (i < cs.size() &&
                           (cs[i]=='-'||cs[i]=='+'||(cs[i]>='0'&&cs[i]<='9'))) {
                    size_t tok_end = i;
                    while (tok_end < cs.size() &&
                           !std::isspace((unsigned char)cs[tok_end]) &&
                           cs[tok_end] != ']') tok_end++;
                    try {
                        if (std::stof(cs.substr(i, tok_end - i)) < -200) arr += ' ';
                    } catch (...) {}
                    i = tok_end;
                } else {
                    i++;
                }
            }
            if (i < cs.size()) i++;
            i = skip_ws(i);
            if (i + 1 < cs.size() && cs[i] == 'T' && cs[i+1] == 'J') {
                out += arr + ' '; i += 2;
            }

        } else if (i + 1 < cs.size() && cs[i] == 'T' &&
                   (cs[i+1]=='d'||cs[i+1]=='D'||cs[i+1]=='m'||cs[i+1]=='*')) {
            out += '\n'; i += 2;

        } else if (i + 1 < cs.size() && cs[i] == 'B' && cs[i+1] == 'T') {
            i += 2;

        } else if (i + 1 < cs.size() && cs[i] == 'E' && cs[i+1] == 'T') {
            out += '\n'; i += 2;

        } else {
            i++;
        }
    }
}

}

std::string langX::pdfExtractText(const std::filesystem::path& pdf_path) {
    std::ifstream f(pdf_path, std::ios::binary);
    if (!f) {
        std::cerr << "[LxPDF] Cannot open: " << pdf_path << "\n";
        return {};
    }
    std::string raw((std::istreambuf_iterator<char>(f)), {});

    {
        int cid_count = 0;
        for (size_t p = 0; (p = raw.find("/CIDFont", p)) != std::string::npos; p += 8) cid_count++;

        if (cid_count > 0) {
            bool has_tou = raw.find("/ToUnicode") != std::string::npos;
            std::cerr << "[LxPDF] WARNING: CIDFont encoding detected (" << cid_count << " font reference(s)).\n" << "        LxPDF cannot decode CID-keyed character streams — extracted text\n" << "        will be garbage or empty. This is a known limitation, not a bug.\n";
            if (has_tou) {
                std::cerr << "        /ToUnicode CMap found: text IS recoverable but CMap parsing is\n" << "        not yet implemented. Keyword to research: 'PDF ToUnicode CMap'.\n";
            } else {
                std::cerr << "        No /ToUnicode CMap — characters cannot be mapped to Unicode at all.\n";
            }
            std::cerr << "        FIX: export the PDF to .txt with a proper library, then pass the .txt file:\n" << "          pip install pymupdf\n" << "          python -c \"import fitz; d=fitz.open('f.pdf'); open('f.txt','w').write('\\n'.join(p.get_text() for p in d))\"\n" << "        or:  pdftotext file.pdf file.txt   (poppler-utils)\n";
        }
    }

    std::string result;
    size_t pos = 0;

    while (pos < raw.size()) {
        size_t sk = raw.find("stream", pos);
        if (sk == std::string::npos) break;

        size_t data_start = sk + 6;
        if (data_start < raw.size() && raw[data_start] == '\r') data_start++;
        if (data_start < raw.size() && raw[data_start] == '\n') data_start++;
        else { pos = sk + 1; continue; }

        size_t endsk = raw.find("endstream", data_start);
        if (endsk == std::string::npos) break;

        size_t obj_pos = raw.rfind("\nobj", sk);
        if (obj_pos == std::string::npos) obj_pos = 0;
        std::string dict(raw.data() + obj_pos, sk - obj_pos);

        bool is_flate = dict.find("FlateDecode") != std::string::npos || dict.find("/Fl\n") != std::string::npos || dict.find("/Fl ") != std::string::npos || dict.find("/Fl\r") != std::string::npos;

        if (is_flate) {
            size_t data_end = endsk;
            while (data_end > data_start &&
                   (raw[data_end-1]=='\r' || raw[data_end-1]=='\n')) data_end--;

            std::string decompressed = inflate_flate(
                (const uint8_t*)(raw.data() + data_start), data_end - data_start);

            if (!decompressed.empty())
                collect_text(decompressed, result);
        }

        pos = endsk + 9;
    }

    return result;
}
