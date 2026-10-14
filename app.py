from pathlib import Path
import json
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from inference import class_names, load_classifier, prepare_image, predict, grad_cam, overlay_heatmap

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title='SatVision — Satellite Land-Cover Intelligence', layout='wide')

@st.cache_resource(show_spinner='Loading the selected MobileNetV2 checkpoint…')
def cached_model():
    return load_classifier()

@st.cache_data(show_spinner=False, max_entries=12)
def analyze(content):
    preview, batch = prepare_image(content)
    probabilities = predict(cached_model(), batch)
    return preview, probabilities

@st.cache_data(show_spinner='Computing Grad-CAM…', max_entries=24)
def explain(content, class_index):
    _, batch = prepare_image(content)
    heatmap, diagnostics = grad_cam(cached_model(), batch, class_index)
    return heatmap, diagnostics['layer']

def main():
    st.title('SatVision — Satellite Land-Cover Intelligence')
    st.write('Classify a satellite image patch into one of ten EuroSAT land-cover classes.')
    st.caption('Selected model: MobileNetV2 · Held-out test accuracy: 93.28% · Macro F1: 0.9292')
    st.warning('Trained on EuroSAT RGB imagery. Predictions on unrelated images may be unreliable. '
               'Softmax probabilities are not calibrated confidence.')
    names = class_names()
    examples = json.loads((ROOT / 'examples/manifest.json').read_text())
    with st.sidebar:
        st.header('Choose an image')
        source = st.radio('Image source', ['Example image', 'Upload image'])
        content = None
        if source == 'Example image':
            selected = st.selectbox('EuroSAT example', range(len(examples)),
                                    format_func=lambda i: examples[i]['label'])
            example = examples[selected]
            content = (ROOT / 'examples' / example['file']).read_bytes()
            st.write('Dataset label:', example['label'])
            st.caption('An illustration, not a new accuracy measurement.')
        else:
            upload = st.file_uploader('JPG, JPEG or PNG', type=['jpg', 'jpeg', 'png'], max_upload_size=10)
            if upload is not None:
                content = upload.getvalue()
        run = st.button('Run classification', type='primary', disabled=content is None)
        st.divider()
        st.caption('EuroSAT examples: Helber et al. (2019), derived from Copernicus Sentinel-2. '
                   'Source and license are included in examples/ATTRIBUTION.md.')
    if content is None:
        st.info('Choose an example or upload an image to begin.')
        return
    try:
        preview, _ = prepare_image(content)
    except ValueError as exc:
        st.error(str(exc)); return
    left, right = st.columns([1, 1.2])
    with left:
        st.subheader('Input patch')
        st.image(preview, width=384)
        st.caption(f'{preview.width} × {preview.height} pixels. Classification uses a 64 × 64 RGB input.')
    # Preserve the result when a user adjusts explanation controls.
    if run:
        st.session_state['classified_content'] = content
    if st.session_state.get('classified_content') == content:
        try:
            _, probabilities = analyze(content)
            with right:
                order = np.argsort(probabilities)[::-1]
                st.subheader(f'Prediction: {names[order[0]]}')
                st.write('Top three softmax probabilities')
                top = pd.DataFrame({'Class': [names[i] for i in order[:3]],
                                    'Probability': [f'{probabilities[i]:.2%}' for i in order[:3]]})
                st.dataframe(top, hide_index=True, width='stretch')
                chart = alt.Chart(pd.DataFrame({'Class': names, 'Probability': probabilities})).mark_bar(
                    color='#246B5A').encode(
                        x=alt.X('Probability:Q', scale=alt.Scale(domain=[0, 1]),
                                axis=alt.Axis(format='.0%'), title='Softmax probability'),
                        y=alt.Y('Class:N', sort=names, title=None),
                        tooltip=['Class:N', alt.Tooltip('Probability:Q', format='.2%')])
                st.altair_chart(chart, width='stretch')
            st.subheader('Grad-CAM explanation')
            target = st.selectbox('Explain a class', range(10), index=int(order[0]),
                                  format_func=lambda i: names[i])
            alpha = st.slider('Overlay transparency (heatmap opacity)', 0.0, 1.0, 0.4, 0.05)
            heatmap, layer = explain(content, target)
            if not np.any(heatmap):
                st.info('This class has no positive Grad-CAM response at the selected feature layer.')
            st.image(overlay_heatmap(preview, heatmap, alpha), width=384)
            st.caption(f'Layer: {layer}. The native heatmap is 3 × 3 and is interpolated for display. '
                       'Grad-CAM is a qualitative explanation, not proof that highlighted regions caused the prediction.')
        except Exception as exc:
            st.error(f'Classification or explanation could not complete: {exc}')
    with st.expander('Dataset, model and limitations', expanded=False):
        st.write('EuroSAT RGB contains 27,000 Sentinel-2 image patches in ten classes. '
                 'The frozen ImageNet-pretrained MobileNetV2 backbone feeds a trained classification head. '
                 'The checkpoint includes its 96 × 96 resize and [-1, 1] scaling. '
                 'The app supplies raw 64 × 64 RGB pixels without a second normalization.')
        st.write('This is patch classification, not pixel segmentation or a monitoring platform. '
                 'The random stratified split does not establish geographic independence. '
                 'Roads, rivers and similar crop textures remain sources of errors. '
                 'Temporal change detection is future work and requires aligned multi-date satellite imagery.')

if __name__ == '__main__':
    main()
