import streamlit as st
from googletrans import Translator
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
st.title('Text Translator')
f={
    'Arabic': 'ar_AR',
    'Czech': 'cs_CZ',
    'German': 'de_DE',
    'English': 'en_XX',
    'Spanish': 'es_XX',
    'Estonian': 'et_EE',
    'Finnish': 'fi_FI',
    'French': 'fr_XX',
    'Gujarati': 'gu_IN',
    'Hindi': 'hi_IN',
    'Italian': 'it_IT',
    'Japanese': 'ja_XX',
    'Kazakh': 'kk_KZ',
    'Korean': 'ko_KR',
    'Lithuanian': 'lt_LT',
    'Latvian': 'lv_LV',
    'Burmese': 'my_MM',
    'Nepali': 'ne_NP',
    'Dutch': 'nl_XX',
    'Romanian': 'ro_RO',
    'Russian': 'ru_RU',
    'Sinhala': 'si_LK',
    'Turkish': 'tr_TR',
    'Vietnamese': 'vi_VN',
    'Chinese': 'zh_CN',
    'Afrikaans': 'af_ZA',
    'Azerbaijani': 'az_AZ',
    'Bengali': 'bn_IN',
    'Persian': 'fa_IR',
    'Hebrew': 'he_IL',
    'Croatian': 'hr_HR',
    'Indonesian': 'id_ID',
    'Georgian': 'ka_GE',
    'Khmer': 'km_KH',
    'Macedonian': 'mk_MK',
    'Malayalam': 'ml_IN',
    'Mongolian': 'mn_MN',
    'Marathi': 'mr_IN',
    'Polish': 'pl_PL',
    'Pashto': 'ps_AF',
    'Portuguese': 'pt_XX',
    'Swedish': 'sv_SE',
    'Swahili': 'sw_KE',
    'Tamil': 'ta_IN',
    'Telugu': 'te_IN',
    'Thai': 'th_TH',
    'Tagalog': 'tl_XX',
    'Ukrainian': 'uk_UA',
    'Urdu': 'ur_PK',
    'Xhosa': 'xh_ZA',
    'Galician': 'gl_ES',
    'Slovene': 'sl_SI'
}
src_lang= st.selectbox(
   "Enter your source language ",
   ['Arabic', 'Czech', 'German', 'English', 'Spanish', 'Estonian', 'Finnish', 'French', 'Gujarati', 'Hindi', 'Italian', 'Japanese', 'Kazakh', 'Korean', 'Lithuanian', 'Latvian', 'Burmese', 'Nepali', 'Dutch', 'Romanian', 'Russian', 'Sinhala', 'Turkish', 'Vietnamese', 'Chinese', 'Afrikaans', 'Azerbaijani', 'Bengali', 'Persian', 'Hebrew', 'Croatian', 'Indonesian', 'Georgian', 'Khmer', 'Macedonian', 'Malayalam', 'Mongolian', 'Marathi', 'Polish', 'Pashto', 'Portuguese', 'Swedish', 'Swahili', 'Tamil', 'Telugu', 'Thai', 'Tagalog', 'Ukrainian', 'Urdu', 'Xhosa', 'Galician', 'Slovene']
,
   index=None,
   placeholder="Select source lanuage",
)
trg_lang= st.selectbox(
   "Enter your target language ",
   ['Arabic', 'Czech', 'German', 'English', 'Spanish', 'Estonian', 'Finnish', 'French', 'Gujarati', 'Hindi', 'Italian', 'Japanese', 'Kazakh', 'Korean', 'Lithuanian', 'Latvian', 'Burmese', 'Nepali', 'Dutch', 'Romanian', 'Russian', 'Sinhala', 'Turkish', 'Vietnamese', 'Chinese', 'Afrikaans', 'Azerbaijani', 'Bengali', 'Persian', 'Hebrew', 'Croatian', 'Indonesian', 'Georgian', 'Khmer', 'Macedonian', 'Malayalam', 'Mongolian', 'Marathi', 'Polish', 'Pashto', 'Portuguese', 'Swedish', 'Swahili', 'Tamil', 'Telugu', 'Thai', 'Tagalog', 'Ukrainian', 'Urdu', 'Xhosa', 'Galician', 'Slovene']
,
   index=None,
   placeholder="Select your target language",
)
text=st.text_area('Enter the Text to Translate')

if st.button('Translate'):
    if text:
        tokenizer.src_lang = f[src_lang]
        encoded_hi = tokenizer(text, return_tensors="pt")
        generated_tokens = model.generate(**encoded_hi,forced_bos_token_id=tokenizer.lang_code_to_id[f[trg_lang]])
        st.success(tokenizer.batch_decode(generated_tokens, skip_special_tokens=True))
    else:
        st.warning('enter your text')



