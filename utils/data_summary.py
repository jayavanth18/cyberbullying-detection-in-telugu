# utils/data_summary.py
"""
Data summary dashboard for the Cyberbullying project.

This module:
- Loads a CSV path supplied by caller
- Shows basic dataset metrics and charts
- Builds a proper Telugu+English WordCloud using token extraction and generate_from_frequencies()
- Is defensive: checks for missing columns, empty data, missing font, etc.
"""

import os
import re
import unicodedata
from collections import Counter

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# WordCloud is optional; guard import
try:
    from wordcloud import WordCloud
    _HAS_WORDCLOUD = True
except Exception:
    _HAS_WORDCLOUD = False

# -----------------------------------------------------------------------------


def find_telugu_font():
    """
    Try common font locations and project fonts for a Telugu-capable TTF.
    Returns path or None.
    """
    candidates = [
        # Windows common Telugu-supporting fonts
        r"C:\Windows\Fonts\Nirmala.ttf",
        r"C:\Windows\Fonts\NirmalaUI.ttf",
        # Project-supplied font locations (recommended)
        os.path.join("data", "fonts", "NotoSansTelugu-Regular.ttf"),
        os.path.join("data", "NotoSansTelugu-Regular.ttf"),
        # Linux common path
        "/usr/share/fonts/truetype/noto/NotoSansTelugu-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansTeluguUI-Regular.ttf",
        # macOS supplemental path
        "/System/Library/Fonts/Supplemental/Nirmala.ttf",
    ]
    for p in candidates:
        try:
            if p and os.path.exists(p):
                return p
        except Exception:
            continue
    return None


@st.cache_data(show_spinner=False)
def safe_read_csv(path):
    """Read CSV defensively and return DataFrame or None."""
    try:
        df = pd.read_csv(path)
        return df
    except FileNotFoundError:
        return None
    except Exception:
        # sometimes pandas fails for weird encodings; try with latin1
        try:
            df = pd.read_csv(path, encoding="latin1")
            return df
        except Exception:
            return None


def render_data_summary_ui(csv_file: str):
    st.title("📊 Data Summary Dashboard")
    st.markdown("### Comprehensive dataset exploration and statistical analysis")
    st.markdown("---")

    # 1) Load dataset safely
    df = safe_read_csv(csv_file)
    if df is None:
        st.error(f"No dataset found at `{csv_file}` or file could not be parsed. Please check the path and encoding.")
        return

    if df is None or df.empty:
        st.warning("The dataset is empty. Please add data through the Data Collection module.")
        return

    st.markdown(f"### Total Records: {df.shape[0]}")
    st.markdown("---")

    # ---------- Toxic flag distribution ----------
    if 'Toxic_flag' in df.columns:
        st.subheader("Toxic vs Non-Toxic Distribution")
        try:
            tc = df['Toxic_flag'].value_counts()
            sizes = [int(tc.get(False, 0)), int(tc.get(True, 0))]
            labels = ['Non-Toxic', 'Toxic']
            if sum(sizes) == 0:
                st.info("No labeled entries found in `Toxic_flag`.")
            else:
                fig1, ax1 = plt.subplots()
                ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, textprops={'color': 'w'})
                ax1.axis('equal')
                # Improve contrast for both themes by drawing a white circle behind labels
                st.pyplot(fig1)
                plt.close(fig1)
        except Exception as e:
            st.warning(f"Could not plot distribution: {e}")
    else:
        st.info("`Toxic_flag` column not found. Some visualizations require this column.")

    st.markdown("---")

    # ---------- Toxic type distribution ----------
    st.subheader("Toxic Type Distribution (Only Toxic Entries)")
    if 'Toxic_flag' in df.columns and df['Toxic_flag'].any():
        toxic_df = df[df['Toxic_flag'] == True]
        if 'Toxic_type' in toxic_df.columns and not toxic_df.empty:
            try:
                fig2, ax2 = plt.subplots(figsize=(10, 5))
                sns.countplot(data=toxic_df, x='Toxic_type',
                              order=toxic_df['Toxic_type'].value_counts().index, ax=ax2)
                plt.xticks(rotation=45)
                st.pyplot(fig2)
                plt.close(fig2)
            except Exception as e:
                st.info(f"Could not render Toxic_type chart: {e}")
        else:
            st.info("No toxic entries to display or `Toxic_type` column missing.")
    else:
        st.info("No toxic entries found (or `Toxic_flag` missing).")

    st.markdown("---")

    # ---------- Language Distribution ----------
    st.subheader("Language Distribution")
    if 'lang' in df.columns:
        try:
            fig3, ax3 = plt.subplots()
            sns.countplot(data=df, x='lang', order=df['lang'].value_counts().index, ax=ax3)
            st.pyplot(fig3)
            plt.close(fig3)
        except Exception as e:
            st.info(f"Could not render language chart: {e}")
    else:
        st.info("'lang' column not present in dataset.")

    st.markdown("---")

    # ---------- WordCloud for Toxic Texts ----------
    st.subheader("WordCloud of Toxic Texts")
    if 'Text' not in df.columns:
        st.info("Dataset does not contain a `Text` column. WordCloud unavailable.")
    else:
        # prioritize toxic texts if available
        if 'Toxic_flag' in df.columns and df['Toxic_flag'].any():
            toxic_df = df[df['Toxic_flag'] == True]
        else:
            toxic_df = df

        if toxic_df.empty:
            st.info("No toxic data available for WordCloud.")
        else:
            toxic_text = " ".join(toxic_df['Text'].astype(str))

            # normalize
            try:
                toxic_text_norm = unicodedata.normalize("NFC", toxic_text)
            except Exception:
                toxic_text_norm = toxic_text

            # tokenization: Telugu sequences OR latin words
            token_pattern = re.compile(r'[\u0C00-\u0C7F]+|[A-Za-z]+', flags=re.UNICODE)
            raw_tokens = token_pattern.findall(toxic_text_norm)

            # clean tokens
            clean_tokens = []
            for t in raw_tokens:
                # strip common punctuation and zero-width marks
                t = t.strip(".,!?;:\"'()-—\u200c\u200b\u200d")
                if not t:
                    continue
                if re.fullmatch(r'[A-Za-z]+', t):
                    t = t.lower()
                if len(t) <= 1:
                    continue
                clean_tokens.append(t)

            # stopwords (basic)
            EN_STOP = {"please", "stop", "and", "the", "to", "of", "is", "in", "this", "that"}
            TELUGU_STOP = {"ra", "ga", "nee", "lo", "la", "aa", "raaa"}

            filtered_tokens = [t for t in clean_tokens if (t not in EN_STOP and t not in TELUGU_STOP)]

            freq_counter = Counter(filtered_tokens)

            min_freq = st.sidebar.number_input("Min token frequency for WordCloud", min_value=1, max_value=20, value=2, step=1)
            freq = {w: c for w, c in freq_counter.items() if c >= min_freq}

            if st.sidebar.checkbox("Show top tokens (debug)"):
                top_n = st.sidebar.number_input("Top N tokens to show", min_value=5, max_value=200, value=30)
                st.write(sorted(freq.items(), key=lambda kv: kv[1], reverse=True)[:top_n])

            if not freq:
                st.info("No words available for WordCloud after filtering. Try lowering the minimum frequency or stopwords.")
            else:
                font_path = find_telugu_font()
                if font_path is None:
                    st.warning("Telugu-capable font not found. WordCloud may render oddly; place a Telugu TTF in data/fonts/ or install Nirmala/NotoTelugu.")
                    font_path = None  # WordCloud will try default font

                if not _HAS_WORDCLOUD:
                    st.info("wordcloud package not installed. Install `wordcloud` to see the word cloud.")
                else:
                    try:
                        wc = WordCloud(
                            width=1000,
                            height=450,
                            background_color='white',
                            font_path=font_path,
                            max_words=300,
                            scale=2,
                            prefer_horizontal=0.9,
                            collocations=False
                        ).generate_from_frequencies(freq)

                        fig4, ax4 = plt.subplots(figsize=(12, 6))
                        ax4.imshow(wc, interpolation='bilinear')
                        ax4.axis('off')
                        st.pyplot(fig4)
                        plt.close(fig4)
                    except Exception as e:
                        st.error(f"WordCloud generation failed: {e}")

    st.markdown("---")

    # ---------- Basic Text Stats ----------
    st.subheader("Basic Text Statistics")
    try:
        df['text_length'] = df['Text'].astype(str).apply(len)
        st.write(f"**Average Text Length:** {df['text_length'].mean():.2f} characters")
        if df['text_length'].dropna().size > 0:
            try:
                longest_idx = int(df['text_length'].idxmax())
                shortest_idx = int(df['text_length'].idxmin())
                st.write(f"**Longest Text:** {df.loc[longest_idx]['Text']}")
                st.write(f"**Shortest Text:** {df.loc[shortest_idx]['Text']}")
            except Exception:
                st.info("Unable to determine longest/shortest text (possible NaNs or non-unique indexes).")
    except Exception as e:
        st.info(f"Could not compute text statistics: {e}")

    st.markdown("---")

    if st.checkbox("Show Raw Dataset"):
        display_df = df.copy()
        if 'text_length' in display_df.columns:
            display_df = display_df.drop(columns=['text_length'])
        st.dataframe(display_df)
