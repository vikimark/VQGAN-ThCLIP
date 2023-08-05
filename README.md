<p align="center">
  <h1 align="center">VQGAN-ThCLIP</h1>
  <h3 align="center">Text-to-image synthesis model in Thai language</h3>
</p>

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vikimark/VQGAN-ThCLIP/blob/master/Streamlit_VQGANxThaiCLIP.ipynb) [![Medium Badge](https://img.shields.io/badge/-How_Thai_image_generative_model_is_made-gray?style=flat-square&labelColor=000000&logo=Medium&link=https://medium.com/@phuritsiritip)](https://medium.com/@phuritsiritip/โครงการ-ai-builders-กับ-ai-สร้างภาพจากข้อความสร้างโดยเด็กมัธยมปลาย-ที่เกือบจะขึ้นปี-1-ed5878c7a72c)

## Overview

This model is text-to-image synthesis based on VQGAN with CLIP as a loss function.<br />CLIP was trained on GCC+MSCOCO sample for 2M sentences, which were translated to Thai by AIReseach translation model, using Teacher learning method by [FreddeFrallan](https://github.com/FreddeFrallan/Multilingual-CLIP)

You can learn more about this project [here!](https://medium.com/@phuritsiritip/โครงการ-ai-builders-กับ-ai-สร้างภาพจากข้อความสร้างโดยเด็กมัธยมปลาย-ที่เกือบจะขึ้นปี-1-ed5878c7a72c)

## Demo

A local deploy streamlit app on Google Colab can be found [here!](https://colab.research.google.com/github/vikimark/VQGAN-ThCLIP/blob/master/Streamlit_VQGANxThaiCLIP.ipynb) <br />Just select Runtime -> run all. wait about 3 minutes, then your url should appear at the bottom of the notebook with the message "your url is ..." (Google Chrome is recommeneded to open the url.)

## Examples

<img src="./sample_image/1_Kc2dl0cYk-K7IY3Nx-k61w.png"></img>

## other relate repo(s)

* [Thai-Cross-CLIP](https://github.com/vikimark/Thai-Cross-CLIP): Thai CLIP text encoder model trained via Teacher Learning

## Acknowledgements

* [AI Builders](https://github.com/ai-builders/ai-builders.github.io) for providing knowledge and support along the way<br />
* [Multilingual-CLIP](https://github.com/FreddeFrallan/Multilingual-CLIP) for Teacher learning method<br />
* [OpenAI's CLIP](https://github.com/openai/CLIP)<br />
* [AIResearch's translation model](https://airesearch.in.th/releases/machine-translation-models)<br />
