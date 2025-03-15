// -----------------------------------------------------------------------------------------
// QSVEnc/NVEnc by rigaya
// -----------------------------------------------------------------------------------------
// The MIT License
//
// Copyright (c) 2019 rigaya
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//
// --------------------------------------------------------------------------------------------


#include <string>
#include <algorithm>
#include "rgy_language.h"

struct RGYLang {
    const char *code_3letter_6392b;
    const char *code_3letter_6392t;
    const char *code_2letter_6391;
    const char *desc;
};

static const RGYLang RGY_LANG_TABLE[] = {
    {"aar", nullptr, "aa", "Afar"},
    {"abk", nullptr, "ab", "Abkhazian"},
    {"afr", nullptr, "af", "Afrikaans"},
    {"aka", nullptr, "ak", "Akan"},
    {"alb", "sqi", "sq", "Albanian"},
    {"amh", nullptr, "am", "Amharic"},
    {"ara", nullptr, "ar", "Arabic"},
    {"arg", nullptr, "an", "Aragonese"},
    {"arm", "hye", "hy", "Armenian"},
    {"asm", nullptr, "as", "Assamese"},
    {"ava", nullptr, "av", "Avaric"},
    {"ave", nullptr, "ae", "Avestan"},
    {"aym", nullptr, "ay", "Aymara"},
    {"aze", nullptr, "az", "Azerbaijani"},
    {"bak", nullptr, "ba", "Bashkir"},
    {"bam", nullptr, "bm", "Bambara"},
    {"baq", "eus", "eu", "Basque"},
    {"bel", nullptr, "be", "Belarusian"},
    {"ben", nullptr, "bn", "Bengali"},
    {"bih", nullptr, "bh", "Bihari languages"},
    {"bis", nullptr, "bi", "Bislama"},
    {"bos", nullptr, "bs", "Bosnian"},
    {"bre", nullptr, "br", "Breton"},
    {"bul", nullptr, "bg", "Bulgarian"},
    {"bur", "mya", "my", "Burmese"},
    {"cat", nullptr, "ca", "Catalan"},
    {"cha", nullptr, "ch", "Chamorro"},
    {"che", nullptr, "ce", "Chechen"},
    {"chi", "zho", "zh", "Chinese"},
    {"chu", nullptr, "cu", "Church Slavic"},
    {"chv", nullptr, "cv", "Chuvash"},
    {"cor", nullptr, "kw", "Cornish"},
    {"cos", nullptr, "co", "Corsican"},
    {"cre", nullptr, "cr", "Cree"},
    {"cze", "ces", "cs", "Czech"},
    {"dan", nullptr, "da", "Danish"},
    {"div", nullptr, "dv", "Divehi"},
    {"dut", "nld", "nl", "Dutch"},
    {"dzo", nullptr, "dz", "Dzongkha"},
    {"eng", nullptr, "en", "English"},
    {"epo", nullptr, "eo", "Esperanto"},
    {"est", nullptr, "et", "Estonian"},
    {"ewe", nullptr, "ee", "Ewe"},
    {"fao", nullptr, "fo", "Faroese"},
    {"fij", nullptr, "fj", "Fijian"},
    {"fin", nullptr, "fi", "Finnish"},
    {"fre", "fra", "fr", "French"},
    {"fry", nullptr, "fy", "Western Frisian"},
    {"ful", nullptr, "ff", "Fulah"},
    {"geo", "kat", "ka", "Georgian"},
    {"ger", "deu", "de", "German"},
    {"gla", nullptr, "gd", "Gaelic"},
    {"gle", nullptr, "ga", "Irish"},
    {"glg", nullptr, "gl", "Galician"},
    {"glv", nullptr, "gv", "Manx"},
    {"gre", "ell", "el", "Greek"},
    {"grn", nullptr, "gn", "Guarani"},
    {"guj", nullptr, "gu", "Gujarati"},
    {"hat", nullptr, "ht", "Haitian"},
    {"hau", nullptr, "ha", "Hausa"},
    {"heb", nullptr, "he", "Hebrew"},
    {"her", nullptr, "hz", "Herero"},
    {"hin", nullptr, "hi", "Hindi"},
    {"hmo", nullptr, "ho", "Hiri Motu"},
    {"hrv", nullptr, "hr", "Croatian"},
    {"hun", nullptr, "hu", "Hungarian"},
    {"ibo", nullptr, "ig", "Igbo"},
    {"ice", "isl", "is", "Icelandic"},
    {"ido", nullptr, "io", "Ido"},
    {"iii", nullptr, "ii", "Sichuan Yi"},
    {"iku", nullptr, "iu", "Inuktitut"},
    {"ile", nullptr, "ie", "Interlingue"},
    {"ina", nullptr, "ia", "Interlingua"},
    {"ind", nullptr, "id", "Indonesian"},
    {"ipk", nullptr, "ik", "Inupiaq"},
    {"ita", nullptr, "it", "Italian"},
    {"jav", nullptr, "jv", "Javanese"},
    {"jpn", nullptr, "ja", "Japanese"},
    {"kal", nullptr, "kl", "Kalaallisut"},
    {"kan", nullptr, "kn", "Kannada"},
    {"kas", nullptr, "ks", "Kashmiri"},
    {"kau", nullptr, "kr", "Kanuri"},
    {"kaz", nullptr, "kk", "Kazakh"},
    {"khm", nullptr, "km", "Central Khmer"},
    {"kik", nullptr, "ki", "Kikuyu"},
    {"kin", nullptr, "rw", "Kinyarwanda"},
    {"kir", nullptr, "ky", "Kirghiz"},
    {"kom", nullptr, "kv", "Komi"},
    {"kon", nullptr, "kg", "Kongo"},
    {"kor", nullptr, "ko", "Korean"},
    {"kua", nullptr, "kj", "Kuanyama"},
    {"kur", nullptr, "ku", "Kurdish"},
    {"lao", nullptr, "lo", "Lao"},
    {"lat", nullptr, "la", "Latin"},
    {"lav", nullptr, "lv", "Latvian"},
    {"lim", nullptr, "li", "Limburgan"},
    {"lin", nullptr, "ln", "Lingala"},
    {"lit", nullptr, "lt", "Lithuanian"},
    {"ltz", nullptr, "lb", "Luxembourgish"},
    {"lub", nullptr, "lu", "Luba-Katanga"},
    {"lug", nullptr, "lg", "Ganda"},
    {"mac", "mkd", "mk", "Macedonian"},
    {"mah", nullptr, "mh", "Marshallese"},
    {"mal", nullptr, "ml", "Malayalam"},
    {"mao", "mri", "mi", "Maori"},
    {"mar", nullptr, "mr", "Marathi"},
    {"may", "msa", "ms", "Malay"},
    {"mlg", nullptr, "mg", "Malagasy"},
    {"mlt", nullptr, "mt", "Maltese"},
    {"mon", nullptr, "mn", "Mongolian"},
    {"nau", nullptr, "na", "Nauru"},
    {"nav", nullptr, "nv", "Navajo"},
    {"nbl", nullptr, "nr", "South Ndebele"},
    {"nde", nullptr, "nd", "North Ndebele"},
    {"ndo", nullptr, "ng", "Ndonga"},
    {"nep", nullptr, "ne", "Nepali"},
    {"nno", nullptr, "nn", "Norwegian Nynorsk"},
    {"nob", nullptr, "nb", "Norwegian Bokmal"},
    {"nor", nullptr, "no", "Norwegian"},
    {"nya", nullptr, "ny", "Chichewa"},
    {"oci", nullptr, "oc", "Occitan (post 1500)"},
    {"oji", nullptr, "oj", "Ojibwa"},
    {"ori", nullptr, "or", "Oriya"},
    {"orm", nullptr, "om", "Oromo"},
    {"oss", nullptr, "os", "Ossetian"},
    {"pan", nullptr, "pa", "Panjabi"},
    {"per", "fas", "fa", "Persian"},
    {"pli", nullptr, "pi", "Pali"},
    {"pol", nullptr, "pl", "Polish"},
    {"por", nullptr, "pt", "Portuguese"},
    {"pus", nullptr, "ps", "Pushto"},
    {"que", nullptr, "qu", "Quechua"},
    {"roh", nullptr, "rm", "Romansh"},
    {"rum", "ron", "ro", "Romanian"},
    {"run", nullptr, "rn", "Rundi"},
    {"rus", nullptr, "ru", "Russian"},
    {"sag", nullptr, "sg", "Sango"},
    {"san", nullptr, "sa", "Sanskrit"},
    {"sin", nullptr, "si", "Sinhala"},
    {"slo", "slk", "sk", "Slovak"},
    {"slv", nullptr, "sl", "Slovenian"},
    {"sme", nullptr, "se", "Northern Sami"},
    {"smo", nullptr, "sm", "Samoan"},
    {"sna", nullptr, "sn", "Shona"},
    {"snd", nullptr, "sd", "Sindhi"},
    {"som", nullptr, "so", "Somali"},
    {"sot", nullptr, "st", "Southern Sotho"},
    {"spa", nullptr, "es", "Spanish"},
    {"srd", nullptr, "sc", "Sardinian"},
    {"srp", nullptr, "sr", "Serbian"},
    {"ssw", nullptr, "ss", "Swati"},
    {"sun", nullptr, "su", "Sundanese"},
    {"swa", nullptr, "sw", "Swahili"},
    {"swe", nullptr, "sv", "Swedish"},
    {"tah", nullptr, "ty", "Tahitian"},
    {"tam", nullptr, "ta", "Tamil"},
    {"tat", nullptr, "tt", "Tatar"},
    {"tel", nullptr, "te", "Telugu"},
    {"tgk", nullptr, "tg", "Tajik"},
    {"tgl", nullptr, "tl", "Tagalog"},
    {"tha", nullptr, "th", "Thai"},
    {"tib", "bod", "bo", "Tibetan"},
    {"tir", nullptr, "ti", "Tigrinya"},
    {"ton", nullptr, "to", "Tonga"},
    {"tsn", nullptr, "tn", "Tswana"},
    {"tso", nullptr, "ts", "Tsonga"},
    {"tuk", nullptr, "tk", "Turkmen"},
    {"tur", nullptr, "tr", "Turkish"},
    {"twi", nullptr, "tw", "Twi"},
    {"uig", nullptr, "ug", "Uighur"},
    {"ukr", nullptr, "uk", "Ukrainian"},
    {"urd", nullptr, "ur", "Urdu"},
    {"uzb", nullptr, "uz", "Uzbek"},
    {"ven", nullptr, "ve", "Venda"},
    {"vie", nullptr, "vi", "Vietnamese"},
    {"vol", nullptr, "vo", "Volapuk"},
    {"wel", "cym", "cy", "Welsh"},
    {"wln", nullptr, "wa", "Walloon"},
    {"wol", nullptr, "wo", "Wolof"},
    {"xho", nullptr, "xh", "Xhosa"},
    {"yid", nullptr, "yi", "Yiddish"},
    {"yor", nullptr, "yo", "Yoruba"},
    {"zha", nullptr, "za", "Zhuang"},
    {"zul", nullptr, "zu", "Zulu"}
};

#pragma warning (push)
#pragma warning (disable: 4244)
#pragma warning (disable: 4996)
static inline std::string rgy_lang_tolowercase(const std::string &str) {
    std::string str_copy = str;
    std::transform(str_copy.cbegin(), str_copy.cend(), str_copy.begin(), tolower);
    return str_copy;
}
#pragma warning (pop)

static int rgy_lang_index(const std::string &lang_code) {
    const std::string lang_code_lower = rgy_lang_tolowercase(lang_code);
    if (lang_code_lower.length() == 2) {
        for (int i = 0; i < (int)(sizeof(RGY_LANG_TABLE) / sizeof(RGY_LANG_TABLE[0])); i++) {
            if (lang_code_lower == RGY_LANG_TABLE[i].code_2letter_6391) {
                return i;
            }
        }
    } else if (lang_code_lower.length() == 3) {
        for (int i = 0; i < (int)(sizeof(RGY_LANG_TABLE) / sizeof(RGY_LANG_TABLE[0])); i++) {
            if (lang_code_lower == RGY_LANG_TABLE[i].code_3letter_6392b ||
                (RGY_LANG_TABLE[i].code_3letter_6392t && lang_code_lower == RGY_LANG_TABLE[i].code_3letter_6392t)) {
                return i;
            }
        }
    }
    return -1;
}

std::string rgy_lang_2letter_6391(const std::string &lang_code) {
    const int idx = rgy_lang_index(lang_code);
    return (idx >= 0) ? RGY_LANG_TABLE[idx].code_2letter_6391 : "";
}
std::string rgy_lang_3letter_6392b(const std::string &lang_code) {
    const int idx = rgy_lang_index(lang_code);
    return (idx >= 0) ? RGY_LANG_TABLE[idx].code_3letter_6392b : "";
}
std::string rgy_lang_desc(const std::string &lang_code) {
    const int idx = rgy_lang_index(lang_code);
    return (idx >= 0) ? RGY_LANG_TABLE[idx].desc : "";
}
bool rgy_lang_equal(const std::string &lang1, const std::string &lang2) {
    const int idx1 = rgy_lang_index(lang1);
    const int idx2 = rgy_lang_index(lang2);
    return (idx1 < 0 || idx2 < 0) ? false : idx1 == idx2;
}
bool rgy_lang_exist(const std::string &lang_code) {
    return rgy_lang_index(lang_code) >= 0;
}

