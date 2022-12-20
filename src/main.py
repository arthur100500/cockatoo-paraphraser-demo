"""
A small streamlit app to demonstrate a Parrot Paraphrasing on T5 Model from Huggingface
Model on HF: https://huggingface.co/prithivida/parrot_paraphraser_on_T5

This is a project for 2022 ML Course practice
"""
import streamlit as st
from parrot_model import ParrotModel


@st.cache(allow_output_mutation=True)
def get_parrot() -> ParrotModel:
    """Generates the parrot model and caches it, as it takes up around 10-15 seconds"""
    return ParrotModel()


def main() -> None:
    """Streamlit app entrypoint"""
    st.set_page_config(
        page_title="Cockatoo",
        page_icon="favicon.ico",
    )

    parrot = get_parrot()

    sidebar = st.sidebar
    sidebar.header("Configure cockatoo")
    sidebar.subheader("Cockatoo parameters")

    parrot.set_seed(int(sidebar.number_input("Seed", 0, 999999, 1)))
    adequacy_threshold = sidebar.slider("Adequacy", 0.0, 1.0, 0.8)
    fluency_threshold = sidebar.slider("Fluency", 0.0, 1.0, 0.8)
    amount = int(sidebar.slider("Maximum amount", 1, 9, 3))
    do_diverse = sidebar.checkbox("Do Diverse?")

    st.title("🦜 Cockatoo paraphraser")
    st.subheader("Type the text you want to paraphrase and get the result!")

    phrase = st.text_area("Input for cockatoo to paraphrase")
    result = parrot.para_phrase(
        phrase,
        do_diverse=do_diverse,
        amount=amount,
        fluency_threshold=fluency_threshold,
        adequacy_threshold=adequacy_threshold,
    )

    if result:
        st.subheader("Cockatoo gave these results")
        for index, value in enumerate(result):
            st.write(f"{str(index + 1)}: {value[0]}")
    else:
        st.subheader("Cockatoo gave no results")


if __name__ == "__main__":
    main()
