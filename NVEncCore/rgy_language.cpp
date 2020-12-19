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
    const char *code_2letter_6391;
    const char *desc;
};

static const RGYLang RGY_LANG_TABLE[] = {
    {"aar","aa","Afar"},
    {"abk","ab","Abkhazian"},
    {"afr","af","Afrikaans"},
    {"aka","ak","Akan"},
    {"alb","sq","Albanian"},
    {"amh","am","Amharic"},
    {"ara","ar","Arabic"},
    {"arg","an","Aragonese"},
    {"arm","hy","Armenian"},
    {"asm","as","Assamese"},
    {"ava","av","Avaric"},
    {"ave","ae","Avestan"},
    {"aym","ay","Aymara"},
    {"aze","az","Azerbaijani"},
    {"bak","ba","Bashkir"},
    {"bam","bm","Bambara"},
    {"baq","eu","Basque"},
    {"bel","be","Belarusian"},
    {"ben","bn","Bengali"},
    {"bih","bh","Bihari languages"},
    {"bis","bi","Bislama"},
    {"bos","bs","Bosnian"},
    {"bre","br","Breton"},
    {"bul","bg","Bulgarian"},
    {"bur","my","Burmese"},
    {"cat","ca","Catalan"},
    {"cha","ch","Chamorro"},
    {"che","ce","Chechen"},
    {"chi","zh","Chinese"},
    {"chu","cu","Church Slavic"},
    {"chv","cv","Chuvash"},
    {"cor","kw","Cornish"},
    {"cos","co","Corsican"},
    {"cre","cr","Cree"},
    {"cze","cs","Czech"},
    {"dan","da","Danish"},
    {"div","dv","Divehi"},
    {"dut","nl","Dutch"},
    {"dzo","dz","Dzongkha"},
    {"eng","en","English"},
    {"epo","eo","Esperanto"},
    {"est","et","Estonian"},
    {"ewe","ee","Ewe"},
    {"fao","fo","Faroese"},
    {"fij","fj","Fijian"},
    {"fin","fi","Finnish"},
    {"fre","fr","French"},
    {"fry","fy","Western Frisian"},
    {"ful","ff","Fulah"},
    {"geo","ka","Georgian"},
    {"ger","de","German"},
    {"gla","gd","Gaelic"},
    {"gle","ga","Irish"},
    {"glg","gl","Galician"},
    {"glv","gv","Manx"},
    {"gre","el","Greek"},
    {"grn","gn","Guarani"},
    {"guj","gu","Gujarati"},
    {"hat","ht","Haitian"},
    {"hau","ha","Hausa"},
    {"heb","he","Hebrew"},
    {"her","hz","Herero"},
    {"hin","hi","Hindi"},
    {"hmo","ho","Hiri Motu"},
    {"hrv","hr","Croatian"},
    {"hun","hu","Hungarian"},
    {"ibo","ig","Igbo"},
    {"ice","is","Icelandic"},
    {"ido","io","Ido"},
    {"iii","ii","Sichuan Yi"},
    {"iku","iu","Inuktitut"},
    {"ile","ie","Interlingue"},
    {"ina","ia","Interlingua"},
    {"ind","id","Indonesian"},
    {"ipk","ik","Inupiaq"},
    {"ita","it","Italian"},
    {"jav","jv","Javanese"},
    {"jpn","ja","Japanese"},
    {"kal","kl","Kalaallisut"},
    {"kan","kn","Kannada"},
    {"kas","ks","Kashmiri"},
    {"kau","kr","Kanuri"},
    {"kaz","kk","Kazakh"},
    {"khm","km","Central Khmer"},
    {"kik","ki","Kikuyu"},
    {"kin","rw","Kinyarwanda"},
    {"kir","ky","Kirghiz"},
    {"kom","kv","Komi"},
    {"kon","kg","Kongo"},
    {"kor","ko","Korean"},
    {"kua","kj","Kuanyama"},
    {"kur","ku","Kurdish"},
    {"lao","lo","Lao"},
    {"lat","la","Latin"},
    {"lav","lv","Latvian"},
    {"lim","li","Limburgan"},
    {"lin","ln","Lingala"},
    {"lit","lt","Lithuanian"},
    {"ltz","lb","Luxembourgish"},
    {"lub","lu","Luba-Katanga"},
    {"lug","lg","Ganda"},
    {"mac","mk","Macedonian"},
    {"mah","mh","Marshallese"},
    {"mal","ml","Malayalam"},
    {"mao","mi","Maori"},
    {"mar","mr","Marathi"},
    {"may","ms","Malay"},
    {"mlg","mg","Malagasy"},
    {"mlt","mt","Maltese"},
    {"mon","mn","Mongolian"},
    {"nau","na","Nauru"},
    {"nav","nv","Navajo"},
    {"nbl","nr","South Ndebele"},
    {"nde","nd","North Ndebele"},
    {"ndo","ng","Ndonga"},
    {"nep","ne","Nepali"},
    {"nno","nn","Norwegian Nynorsk"},
    {"nob","nb","Norwegian Bokmal"},
    {"nor","no","Norwegian"},
    {"nya","ny","Chichewa"},
    {"oci","oc","Occitan (post 1500)"},
    {"oji","oj","Ojibwa"},
    {"ori","or","Oriya"},
    {"orm","om","Oromo"},
    {"oss","os","Ossetian"},
    {"pan","pa","Panjabi"},
    {"per","fa","Persian"},
    {"pli","pi","Pali"},
    {"pol","pl","Polish"},
    {"por","pt","Portuguese"},
    {"pus","ps","Pushto"},
    {"que","qu","Quechua"},
    {"roh","rm","Romansh"},
    {"rum","ro","Romanian"},
    {"run","rn","Rundi"},
    {"rus","ru","Russian"},
    {"sag","sg","Sango"},
    {"san","sa","Sanskrit"},
    {"sin","si","Sinhala"},
    {"slo","sk","Slovak"},
    {"slv","sl","Slovenian"},
    {"sme","se","Northern Sami"},
    {"smo","sm","Samoan"},
    {"sna","sn","Shona"},
    {"snd","sd","Sindhi"},
    {"som","so","Somali"},
    {"sot","st","Southern Sotho"},
    {"spa","es","Spanish"},
    {"srd","sc","Sardinian"},
    {"srp","sr","Serbian"},
    {"ssw","ss","Swati"},
    {"sun","su","Sundanese"},
    {"swa","sw","Swahili"},
    {"swe","sv","Swedish"},
    {"tah","ty","Tahitian"},
    {"tam","ta","Tamil"},
    {"tat","tt","Tatar"},
    {"tel","te","Telugu"},
    {"tgk","tg","Tajik"},
    {"tgl","tl","Tagalog"},
    {"tha","th","Thai"},
    {"tib","bo","Tibetan"},
    {"tir","ti","Tigrinya"},
    {"ton","to","Tonga"},
    {"tsn","tn","Tswana"},
    {"tso","ts","Tsonga"},
    {"tuk","tk","Turkmen"},
    {"tur","tr","Turkish"},
    {"twi","tw","Twi"},
    {"uig","ug","Uighur"},
    {"ukr","uk","Ukrainian"},
    {"urd","ur","Urdu"},
    {"uzb","uz","Uzbek"},
    {"ven","ve","Venda"},
    {"vie","vi","Vietnamese"},
    {"vol","vo","Volapuk"},
    {"wel","cy","Welsh"},
    {"wln","wa","Walloon"},
    {"wol","wo","Wolof"},
    {"xho","xh","Xhosa"},
    {"yid","yi","Yiddish"},
    {"yor","yo","Yoruba"},
    {"zha","za","Zhuang"},
    {"zul","zu","Zulu"}
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
        for (int i = 0; i < _countof(RGY_LANG_TABLE); i++) {
            if (lang_code_lower == RGY_LANG_TABLE[i].code_2letter_6391) {
                return i;
            }
        }
    } else if (lang_code_lower.length() == 3) {
        for (int i = 0; i < _countof(RGY_LANG_TABLE); i++) {
            if (lang_code_lower == RGY_LANG_TABLE[i].code_3letter_6392b) {
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

