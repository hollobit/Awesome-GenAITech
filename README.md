# Awesome Generative AI Techniques

A Classification and a list of papers and other resources on Generative AI Techniques. 

<img src="GAIT-classification.png" width="90%">

Generative AI refers to a type of artificial intelligence, most commonly associated with machine learning, that is used to create content. It's called "generative" because it can generate new, previously unseen data that mirrors the training data.

Generative AI systems can create a wide range of outputs, including but not limited to text, images, music, and voice. 

## Table of Content

## Generative ML models

A few key examples of ML models for generative AI are:

* Generative Adversarial Networks (GANs): This is a class of machine learning frameworks where two neural networks contest with each other. One network, called the "generator", generates new data instances, while the other, the "discriminator", evaluates them for authenticity. The generator improves its ability to create realistic data, and the discriminator enhances its ability to distinguish real data from artificial ones. GANs are often used to generate realistic images, enhance image resolution (Super Resolution), or perform image-to-image translation (changing daytime scenes to nighttime, or changing a horse into a zebra, for example).

* Variational Autoencoders (VAEs): These are generative models that use deep learning techniques to produce sophisticated and compressed representations of input data (encoding), and then generate new data from these representations (decoding). VAEs are often used in tasks that involve generating examples that are variations of the input data, such as creating new images that resemble a training set of images.

* Transformer Models: These models are primarily used in natural language processing to generate text. They work by processing input data (like a sentence or paragraph) in parallel rather than sequentially, allowing for more efficient computation and the ability to handle longer sequences of text. GPT-3, developed by OpenAI, is an example of a transformer model used for text generation.

Generative AI techniques are powerful, but they also raise some ethical and societal concerns, such as the potential for misuse in deepfakes or generating misleading information.

## Generative tasks

Generative tasks in machine learning (ML) refer to the creation of new data samples that are statistically similar to the input data. These tasks utilize generative models, a subset of ML models, which learn the joint probability distribution of the input data, and can then generate new data samples from the learned distribution.

Let's look at some types of generative tasks and the generative models often associated with them:

* Image Generation: Models like Generative Adversarial Networks (GANs) and Variational Autoencoders (VAEs) are trained to generate new images that resemble the ones they were trained on. This can include anything from generating images of faces that don't exist to creating artwork.

* Text Generation: Models like GPT-3, a Transformer-based model, can generate human-like text. Given a prompt or starting sentence, the model generates the rest of the content.

* Speech Synthesis: Speech synthesis involves generating human-like speech from written text. This is commonly used in text-to-speech systems.

* Data Augmentation: Generative models can create synthetic data that resembles the original training data. This is useful when the amount of real data is limited.

* Sequence Generation: Models such as Recurrent Neural Networks (RNNs) and Long Short-Term Memory (LSTM) networks are used for sequence generation tasks, such as generating a sequence of words, music notes, or a series of future stock prices.

* Image-to-Image Translation: GANs are also used for tasks where the goal is to translate one type of image into another, for example, transforming a sketch into a colored image, or changing day scenes into night scenes.

* Super-resolution: Generative models can take a low-resolution image and generate a high-resolution version of the same image.

## text
| Type of Generative AI Technique | Description | Examples of Medical Applications |
|:-:|:--|:--|
|Text-to-Text|	Converts textual descriptions into other forms of text.|	Used to generate medical reports, patient summaries, or transform complex medical information into plain language for patients.	|																						
|Image-to-Text|	Converts image data into textual descriptions.|	Used in radiology to convert images like X-rays or CT scans into radiology reports.|															|Text-to-Image|	Generates image data from text descriptions.|	Could be used to generate educational illustrations based on textual descriptions of anatomical structures or physiological processes.|																							
|Video-to-Text|	Converts video data into textual descriptions.|	Used to generate surgery reports or patient monitoring reports based on video inputs.|														|Text-to-Video|	Generates video data from textual descriptions.	|Could be used to generate educational videos based on descriptions of surgical procedures or medical conditions.|																							
|Audio-to-Text|	Converts audio data into textual descriptions.|	Used in transcribing patient-doctor conversations, telemedicine calls, or dictations for electronic health records.|																							
|Text-to-Audio|	Generates audio data from textual descriptions.|	Used in text-to-speech systems for visually impaired patients or for reading out electronic health records or educational material.	|																						
|BioSignal-to-Text|	Converts biological signals into textual descriptions.|	Used to generate reports or alerts based on bio-signal data like ECG or EEG readings.	|								|Text-to-BioSignal|	Generates biological signal data from textual descriptions.|	Could be used to simulate bio-signal scenarios based on descriptions for education or testing of medical devices.|																							
|Pharmacogenomics-to-Text|	Converts drug response data into textual descriptions.|	Used to generate personalized medication reports based on a patient's genomic information.|		|Text-to-Pharmacogenomics|	Generates drug response data from textual descriptions.|	Could be used to simulate drug response scenarios based on descriptions of patient's genetic information.		|																					
|Epidemiology-to-Text|	Converts epidemiological data into textual descriptions.|	Used to generate public health reports, research summaries, or education material based on epidemiological data.	|																						
|Text-to-Epidemiology|	Generates epidemiological data from text descriptions.|	Could be used to model the spread of a disease or simulate epidemic scenarios based on descriptions.|																							
|GenomicVariations-to-Text|	Converts genomic variation data into text descriptions.|	Used to generate personalized genetic reports or genetic counseling material.	|							|Text-to-GenomicVariations|	Generates genomic variation data from text descriptions.|	Could be used to model patterns of specific genomic variations or simulate scenarios of genomic variations based on text descriptions.	|																			

### Text-to-Text
* [Text Generation](https://paperswithcode.com/task/text-generation)
* [A survey on text generation using generative adversarial networks](https://dl.acm.org/doi/10.1016/j.patcog.2021.108098)
* Pretrained Language Models for Text Generation: A Survey ([:x:](https://arxiv.org/abs/2201.05273)), ([:paperclip:](https://arxiv.org/pdf/2201.05273.pdf)), ([:orange_book:](https://www.arxiv-vanity.com/papers/2201.05273)), ([:house:](https://huggingface.co/papers/2201.05273)), ([:eight_spoked_asterisk:](https://paperswithcode.com/paper/a-survey-of-pretrained-language-models-based)) 
* A Survey of Controllable Text Generation using Transformer-based Pre-trained Language Models ([:x:](https://arxiv.org/abs/2201.05337)), ([:paperclip:](https://arxiv.org/pdf/2201.05337.pdf)), ([:orange_book:](https://www.arxiv-vanity.com/papers/2201.05337)), ([:house:](https://huggingface.co/papers/2201.05337)), ([:eight_spoked_asterisk:](https://paperswithcode.com/paper/a-survey-of-controllable-text-generation)) 
* [Towards User-Centric Text-to-Text Generation: A Survey](https://link.springer.com/chapter/10.1007/978-3-030-83527-9_1)

### Text-to-SQL
* [Text-To-SQL](https://paperswithcode.com/task/text-to-sql) 

### Text-to-Image
* [Text-to-Image Generation](https://paperswithcode.com/task/text-to-image-generation)
* Text-to-image Diffusion Models in Generative AI: A Survey ([:x:](https://arxiv.org/abs/2303.07909)), ([:paperclip:](https://arxiv.org/pdf/2303.07909.pdf)), ([:orange_book:](https://www.arxiv-vanity.com/papers/2303.07909)), ([:house:](https://huggingface.co/papers/2303.07909)), ([:eight_spoked_asterisk:](https://paperswithcode.com/paper/text-to-image-diffusion-model-in-generative)) 
* [A survey on generative adversarial network-based text-to-image synthesis](https://www.sciencedirect.com/science/article/abs/pii/S0925231221006111) 
* [𝓐𝔀𝓮𝓼𝓸𝓶𝓮 𝓣𝓮𝔁𝓽📝-𝓽𝓸-𝓘𝓶𝓪𝓰𝓮🌇](https://github.com/Yutong-Zhou-cv/Awesome-Text-to-Image)![GitHub Repo stars](https://img.shields.io/github/stars/Yutong-Zhou-cv/Awesome-Text-to-Image?style=social))

### Text-to-3D
* [Text to 3D](https://paperswithcode.com/task/text-to-3d)
* Generative AI meets 3D: A Survey on Text-to-3D in AIGC Era ([:x:](https://arxiv.org/abs/2305.06131)), ([:paperclip:](https://arxiv.org/pdf/2305.06131.pdf)), ([:orange_book:](https://www.arxiv-vanity.com/papers/2305.06131)), ([:house:](https://huggingface.co/papers/2305.06131)), ([:eight_spoked_asterisk:](https://paperswithcode.com/paper/generative-ai-meets-3d-a-survey-on-text-to-3d)) 

### Text-to-Audio
* [Audio Generation](https://paperswithcode.com/task/audio-generation)
* [Text to Audio Retrieval](https://paperswithcode.com/task/text-to-audio-retrieval)
* [A Survey on Audio Diffusion Models: Text To Speech Synthesis and Enhancement in Generative AI](https://ui.adsabs.harvard.edu/abs/2023arXiv230313336Z/abstract)

### Text-to-music 
* [Text-to-Music Generation](https://paperswithcode.com/task/text-to-music-generation)

### Text-to-speech
* [Text-To-Speech Synthesis](https://paperswithcode.com/task/text-to-speech-synthesis) 


## image
| Type of Generative AI Technique | Description | Examples of Medical Applications |
|:-:|:--|:--|
|Text-to-Image|	Converts textual descriptions into image data.|	Used to generate medical illustrations from descriptions of anatomical or physiological processes.		|						|Image-to-Text|	Translates image data into textual descriptions.|	Used to create radiology reports from images like X-rays or CT scans.	|																					|Image-to-Image|	Transforms one type of image into another.|	Used in medical imaging to convert between imaging modalities (e.g., MRI to CT) or to enhance image quality.	|			|Video-to-Image|	Converts video data into image data.|	Used to create static images from medical videos for further analysis or documentation.|																	|Image-to-Video|	Generates video data from image inputs.|	Used to create dynamic visualizations from static medical images, such as animated 3D renderings from CT scans.|			|Audio-to-Image|	Converts audio data into image data.|	Used to create visual representations of audio data, such as sonograms from heart sounds.|																|Image-to-Audio|	Generates audio data from image inputs.|	Could be used to sonify medical images, providing an alternative way to interpret the data.		|												|BioSignal-to-Image|	Converts biological signals into image data.|	Used to create visual representations of bio-signal data, such as heat maps of brain activity from EEG signals.	|																						
|Image-to-BioSignal	|Generates biological signal data from image inputs.|	Could be used to simulate bio-signal scenarios based on medical images for device testing or training purposes.		|																					
|Pharmacogenomics-to-Image|	Converts drug response data into images.|	Used to create visual representations of drug responses at the molecular level, such as protein-drug interactions.		|																					
|Image-to-Pharmacogenomics|	Generates drug response data from image inputs.|	Could be used to predict drug responses based on images of cellular reactions or patient-specific medical images.	|																						
|Epidemiology-to-Image|	Converts epidemiological data into images.|	Used to create visual representations of disease spread, such as heat maps or geographic distributions.	|			|Image-to-Epidemiology|	Generates epidemiological data from image inputs.|	Could be used to predict disease spread based on images of social behavior or geographic conditions.	|																						
|GenomicVariations-to-Image|	Converts genomic variation data into images.|	Used to create visual representations of genomic variations, such as images of DNA structures incorporating specific variants.		|																					
|Image-to-GenomicVariations|	Generates genomic variation data from image inputs.|	Could be used to predict genomic variations based on cellular images or images of genetic structures.|																							
|Text-to-X-ray|	Generates X-ray images from textual descriptions.|	Could be used to create educational or simulation scenarios based on textual descriptions of conditions or diseases.	|																						
|X-ray-to-Text|	Translates X-ray images into textual descriptions.|	Used to create radiology reports from X-ray images.	|					|Text-to-MRI|	Generates MRI images from textual descriptions.|	Used for simulation and educational purposes based on textual descriptions of conditions or diseases.|																							
|MRI-to-Text|	Translates MRI images into textual descriptions.|	Used to create radiology reports from MRI images.		|							|MRI-to-CT|	Transforms MRI images into CT images.|	Used for data augmentation or to simulate CT when not available.	|						|Text-to-CT|	Generates CT images from textual descriptions.|	Used for simulation and educational purposes based on textual descriptions of conditions or diseases.	|																						
|CT-to-Text|	Translates CT images into textual descriptions.|	Used to create radiology reports from CT images.|									|Text-to-Ultrasound|	Generates ultrasound images from textual descriptions.|	Could be used for training purposes based on textual descriptions of conditions or diseases.	|																						
|Ultrasound-to-Text|	Translates ultrasound images into textual descriptions.|	Used to create radiology reports from ultrasound images.|																					|Text-to-PET|	Generates PET images from textual descriptions.|	Used for simulation and educational purposes based on textual descriptions of conditions or diseases.|						|PET-to-Text|	Translates PET images into textual descriptions.|	Used to create radiology reports from PET images.	|							
|Text-to-Histopathology|	Generates histopathological images from textual descriptions.|	Could be used for training and education purposes based on descriptions of pathological findings.|																							
|Histopathology-to-Text|	Translates histopathological images into textual descriptions.|	Used to create pathology reports from histopathological images.	|												|Text-to-Microscopy|	Generates microscopy images from textual descriptions.|	Could be used for training and education purposes based on descriptions of microscopic findings.|		|Microscopy-to-Text|	Translates microscopy images into textual descriptions.|	Used to create laboratory reports from microscopic images.	|																			|Text-to-Retinal|	Generates retinal images from textual descriptions.|	Could be used for training and education purposes based on descriptions of ophthalmic conditions.|				|Retinal-to-Text|	Translates retinal images into textual descriptions.|	Used to create ophthalmology reports from retinal images.	|

### Image-to-text
* [Image-to-Text Retrieval](https://paperswithcode.com/task/image-to-text-retrieval]

### Image-to-Image
* [Image-to-Image Translation](https://paperswithcode.com/task/image-to-image-translation)
* [awesome image-to-image translation](https://github.com/weihaox/awesome-image-translation)![GitHub Repo stars](https://img.shields.io/github/stars/weihaox/awesome-image-translation?style=social))
* Image-to-Image Translation: Methods and Applications ([:x:](https://arxiv.org/abs/2101.08629)), ([:paperclip:](https://arxiv.org/pdf/2101.08629.pdf)), ([:orange_book:](https://www.arxiv-vanity.com/papers/2101.08629)), ([:house:](https://huggingface.co/papers/2101.08629)), ([:eight_spoked_asterisk:]()) 
* [Unsupervised Image-to-Image Translation: A Review](https://www.mdpi.com/1424-8220/22/21/8540)

## video
| Type of Generative AI Technique | Description | Examples of Medical Applications |
|:-:|:--|:--|
|Video-to-BioSignal|	Generates biological signal data from video inputs.|	Could be used to predict bio-signals from videos capturing physical responses, such as pulse rate from facial videos.|
|Video-to-Epidemiology|	Generates epidemiological data from video inputs.|	Used to predict the spread of a disease based on social behavior patterns captured in videos.|
|Video-to-GenomicVariations|	Generates genomic variation data from video inputs.|	Used to predict specific genomic variations based on an individual's behavior patterns.|
|Video-to-Microbiomics|	Generates microbial data from video inputs.|	Could be used to monitor changes in microbial communities over time in video footage.|
|Video-to-Pharmacogenomics|	Generates drug response data from video inputs.|	Could be used to monitor the effects of drugs over time in video footage.|
|Video-to-Text|	Generates descriptive text from video data.|	Used for automated note-taking during surgical procedures or other medical events.|
|Video-to-Video|	Converts one style of video into another style.|	Could be used to enhance the clarity of surgical procedure videos or simulate different surgical scenarios.|

## audio
| Type of Generative AI Technique | Description | Examples of Medical Applications |
|:-:|:--|:--|
|Audio-to-BioSignal|	Transforms audio data into bio-signals.|	Could be used to predict heart rate or stress levels based on voice data.|
|Audio-to-Epidemiology|	Generates epidemiological data from audio inputs.|	Used to predict the spread of a disease based on social conversation patterns.|
|Audio-to-GenomicVariations|	Generates genomic variation data from audio inputs.|	Used to predict specific genomic variations based on sound analysis.|
|Audio-to-Microbiomic|s	Generates microbial data from audio inputs.|	Could be used to correlate sounds in an environment with changes in microbial communities.|
|Audio-to-Pharmacogenomics|	Generates drug response data from audio inputs.|	Used to correlate sounds with potential drug responses.|
|Audio-to-Text|	Transcribes audio content into written form.|	Used to transcribe physician-patient conversations, medical lectures, or medical audio books.|

## 3D model
| Type of Generative AI Technique | Description | Examples of Medical Applications |
|:-:|:--|:--|
|Text-to-3D Model|	Converts textual descriptions into 3D models.|	Could be used to create 3D anatomical models for medical education based on textual descriptions.|
|3D Model-to-Text|	Generates descriptive text from 3D model data.|	Used for automated report generation from 3D imaging like CT or MRI scans.|
|Image-to-3D Model|	Converts 2D images into 3D models.|	Used to reconstruct 3D models from 2D medical images like MRI slices.|
|3D Model-to-Image|	Generates 2D images from 3D models.|	Used to generate perspective or cross-sectional images from 3D models of organs or other structures.|
|Video-to-3D Model|	Converts video data into 3D models.|	Could be used to generate 3D models of moving organs (like the heart) based on video footage.|
|3D Model-to-Video|	Creates video content from 3D model data.|	Used to create animated views of 3D anatomical models for medical education or surgical planning.|
|Audio-to-3D Model|	Converts audio data into 3D models.|	Could be used to visualize the vocal tract or other audio-producing structures based on audio inputs.|
|3D Model-to-Audio|	Generates sound from 3D model data.|	Could be used to simulate the sound of a heartbeat or other bodily function based on a 3D model.|
|BioSignal-to-3D Model|	Converts biological signals into 3D models.|	Could be used to create 3D models of bioelectric fields (like those produced by the heart or brain) based on bio-signal inputs.|
|3D Model-to-BioSignal|	Generates biological signal data from 3D model inputs.|	Could be used to predict bio-signals from 3D models of organs or other structures.|
|Pharmacogenomics-to-3D Model|	Converts drug response data into 3D models.	Used to create 3D models of protein-drug interactions based on pharmacogenomic data.|
|3D Model-to-Pharmacogenomics|	Generates drug response data from 3D model inputs.|	Could be used to predict drug responses based on 3D models of patient-specific proteins.|
|Epidemiology-to-3D Model|	Converts epidemiological data into 3D models.|	Used to create 3D visualizations of disease spread in a population.|
|3D Model-to-Epidemiology|	Generates epidemiological data from 3D model inputs.|	Could be used to predict disease spread based on 3D models of population distribution and mobility.|
|GenomicVariations-to-3D Model|	Converts genomic variation data into 3D models.|	Used to create 3D models of genetic structures (like DNA or proteins) that incorporate specific genomic variations.|
|3D Model-to-GenomicVariations|	Generates genomic variation data from 3D model inputs.|	Could be used to predict potential genomic variations based on 3D models of genetic structures.|

## Bio signal
| Type of Generative AI Technique | Description | Examples of Medical Applications |
|:-:|:--|:--|
|BioSignal-to-Audio|	Transforms biological signals into sound.|	Could be used to sonify heartbeats or brain waves for diagnostic purposes.|
|BioSignal-to-Epidemiology|	Generates epidemiological data from biological signals.|	Used to predict the spread of a disease based on ECG or EEG signals.|
|BioSignal-to-GenomicVariations|	Generates genomic variation data from biological signals.|	Used to predict the occurrence of specific genomic variations based on the analysis of bio-signals such as ECG or EEG.|
|BioSignal-to-Image|	Converts biological signals into image data.|	Used to create visual aids for understanding complex bio-signal patterns, like EEG or EKG signals.|
|BioSignal-to-Microbiomics|	Generates microbial data from biological signals.|	Used to correlate physiological signals with changes in microbial communities.|
|BioSignal-to-Pharmacogenomics|	Generates drug response data from biological signals.|	Could be used to predict drug responses based on ECG or EEG signals.|
|BioSignal-to-Text|	Converts biological signal data into textual descriptions.|	Used for automated report generation from ECG or EEG readings.|
|BioSignal-to-Video|	Converts biological signals into video data.|	Used to create simulated videos of physiological events based on bio-signal inputs, such as heartbeat simulations.|

## Pathology
| Type of Generative AI Technique | Description | Examples of Medical Applications |
|:-:|:--|:--|

## Molecular structure
| Type of Generative AI Technique | Description | Examples of Medical Applications |
|:-:|:--|:--|

## Omics
| Type of Generative AI Technique | Description | Examples of Medical Applications |
|:-:|:--|:--|
|GenomicVariations-to-Audio|	Converts genomic variation data into sound.|	Used to convert the changes of specific genomic variations into listenable audio.|
|GenomicVariations-to-GenomicVariations|	Translates one type of genomic variation data into another.|	Used to model how a specific genomic variation might influence another.|
|GenomicVariations-to-Image|	Converts genomic variation data into images.|	Used to visualize the patterns of genomic variations.|
|GenomicVariations-to-Text|	Converts genomic variation data into text descriptions.|	Used to generate diagnosis reports or genetic education materials.|
|GenomicVariations-to-Video|	Converts genomic variation data into video.|	Used to create videos visualizing the changes of specific genomic variations.|
|Microbiomics-to-Audio|	Converts microbial data into sound.|	Could be used to sonify changes in microbial communities for better comprehension.|
|Microbiomics-to-Image|	Converts microbial data into images.|	Used to visualize microbial communities or changes therein.|
|Microbiomics-to-Microbiomics|	Translates one type of microbial data into another.|	Used to predict the effect of certain factors on microbial community composition.|
|Microbiomics-to-Text|	Converts microbial data into text descriptions.|	Used to generate reports or insights about the state of microbial communities.|
|Microbiomics-to-Video|	Converts microbial data into video.|	Could be used to visualize how microbial communities change over time.|
|Pharmacogenomics-to-Audio|	Converts drug response data into sound.|	Could be used to sonify the potential effects of drugs for better comprehension.|
|Pharmacogenomics-to-Image|	Converts drug response data into images.|	Used to visualize the potential effects of drugs based on genetic information.|
|Pharmacogenomics-to-Pharmacogenomics|	Translates one type of drug response data into another.|	Used to predict the response to one drug based on the known response to another drug.|
|Pharmacogenomics-to-Text|	Converts drug response data into textual descriptions.|	Used to generate reports about potential drug responses based on genetic information.|
|Pharmacogenomics-to-Video|	Converts drug response data into video.|	Used to visualize how drug responses could change over time.|

## Health record
| Type of Generative AI Technique | Description | Examples of Medical Applications |
|:-:|:--|:--|
| **Text-to-Health Record** | AI models can translate raw text data (e.g., doctor's notes, patient self-reports) into structured health record data. | Streamlining data entry, improving the comprehensiveness of health records. |
| **Health Record-to-Text** | AI models can generate medical reports, summaries, or explanations based on structured health record data. | Automating documentation, improving patient understanding of their health conditions. |
| **Health Record-to-Image (Visual Representation)** | AI models can create visual representations or infographics based on health record data. | Enhancing patient engagement, supporting healthcare providers in understanding patient history at a glance. |
| **Health Record-to-Video (Simulation)** | AI models can generate a video simulation or timeline of a patient's health trajectory based on historical health record data. | Providing visual aids for patient education, facilitating doctor-patient communication. |

## Claims
| Type of Generative AI Technique | Description | Examples of Medical Applications |
|:-:|:--|:--|

## Issues
Generative AI, while offering numerous possibilities, also presents a range of issues and challenges. Here are some of the major concerns:

1. **Quality of Generated Content:** While Generative AI models can create realistic outputs, the consistency and quality, especially across diverse data inputs, can be variable. For example, the generated text might still occasionally lack context, show bias, or provide inappropriate or nonsensical responses.

2. **Data Privacy and Security:** Generative models typically require large volumes of data for training, which might include private or sensitive information. There's a potential risk to privacy if models unintentionally memorize and reproduce such data.

3. **Ethical Considerations:** There's the potential for misuse of Generative AI in the creation of deepfakes – highly realistic images, videos, or audio of individuals appearing to say or do things that they didn't. This can raise serious ethical and legal issues, as deepfakes can be used for purposes such as misinformation, fraud, or harassment.

4. **Bias:** As with any AI model, generative models can also mirror and amplify the biases present in their training data. A text-generating model, for instance, might produce gender-biased content if trained on a dataset with predominantly male authors.

5. **Computational Resources:** Training generative models can be computationally expensive and contribute to environmental issues due to the high energy consumption of data centers.

6. **Evaluation Challenges:** Measuring the performance of generative models can be difficult. In text or image generation, for example, evaluating the creativity or relevance of generated content in a quantitative manner can be challenging.

7. **Regulatory Challenges:** As with many emerging technologies, there is a lack of clear regulations surrounding the use of generative AI, which can make it difficult to ensure accountability and prevent misuse.


## Contributing
This is an active repository and your contributions are always welcome!

I will keep some pull requests open if I'm not sure if they are awesome for Generative AI Techniques, you could vote for them by adding 👍 to them.

---

If you have any question about this opinionated list, do not hesitate to contact me hollobit@etri.re.kr.
