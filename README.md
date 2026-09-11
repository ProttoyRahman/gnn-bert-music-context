# **GNN-BERT Music Context**

A supervised neural network project exploring how music context can be represented by combining audio structure and text semantics using Graph Neural Networks (GNNs) and BERT.

## **Project Overview**

Music contains multiple forms of information, including melody, harmony, rhythm, metadata, and semantic information. This project investigates whether combining structured audio representations with text representations can improve music-context understanding.

The system uses:

* **GraphSAGE** to represent the structural relationships between segments of a music track.  
* **DistilBERT** to encode text metadata.  
* **GNN-BERT Fusion** to combine audio-graph and text representations.  
* **Contrastive learning with InfoNCE** to learn a shared audio-text embedding space.

## **Dataset**

The implementation uses **FMA-small** from the Free Music Archive dataset.

For this project:

* 800 tracks were selected.  
* 8 genres were used:  
  * Electronic  
  * Experimental  
  * Folk  
  * Hip-Hop  
  * Instrumental  
  * International  
  * Pop  
  * Rock  
* Each track was represented using 30 seconds of audio.  
* Audio was divided into six 5-second segments.

The dataset was divided into:

* 560 training tracks  
* 120 validation tracks  
* 120 test tracks

## **Audio Feature Extraction**

Each 5-second audio segment was represented using:

* 12 chroma features  
* 13 MFCC features

This produced a 25-dimensional feature vector for every graph node.

## **Graph Construction**

Each music track was represented as a graph.

* Each 5-second audio segment represents a node.  
* Temporal edges connect consecutive segments.  
* Additional edges connect segments whose feature vectors have cosine similarity of at least 0.75.

The resulting graph represents both the temporal structure and acoustic similarity of different portions of a track.

## **Task 1 — BERT Text Classification**

DistilBERT (`distilbert-base-uncased`) was used as the text encoder.

The text representation was constructed from available FMA metadata:

* Artist  
* Album  
* Track title

The resulting BERT representation was classified into one of the eight music genres.

## **Task 2 — Graph Neural Network**

A two-layer GraphSAGE model was used to process the music graphs.

Architecture:

25-dimensional node features  
        ↓  
GraphSAGE  
        ↓  
64-dimensional hidden representation  
        ↓  
GraphSAGE  
        ↓  
64-dimensional graph representation  
        ↓  
Global Mean Pooling  
        ↓  
8-class classifier

A CNN operating on mel-spectrogram representations was also implemented as an audio baseline.

## **Task 3 — GNN-BERT Fusion**

The GraphSAGE graph embedding and DistilBERT text embedding were concatenated.

GraphSAGE embedding (64)  
             \+  
DistilBERT embedding (768)  
             ↓  
       Concatenation  
             ↓  
        256 neurons  
             ↓  
        8-class output

The fusion model was evaluated against the individual CNN, GNN, and BERT models.

### **Classification Results**

| Model | Accuracy | Macro-F1 | Micro-F1 |
| ----- | ----- | ----- | ----- |
| CNN Mel Baseline | 18.33% | 0.0855 | 18.33% |
| GNN GraphSAGE | 36.67% | 0.3330 | 36.67% |
| BERT-only | 78.33% | 0.7806 | 78.33% |
| GNN-BERT Fusion | 75.00% | 0.7459 | 75.00% |

## **Task 4 — Contrastive Audio-Text Learning**

A dual-encoder architecture was implemented to learn a shared representation between music graphs and text.

The GraphSAGE and DistilBERT representations were projected into a shared 128-dimensional embedding space.

InfoNCE contrastive loss was used to make the corresponding audio and text representations similar while treating other samples in the batch as negative pairs.

The system was evaluated using cross-modal retrieval:

* Audio → Text  
* Text → Audio

using Recall@1, Recall@5, and Recall@10.

### **Contrastive Retrieval Results**

| Direction | Recall@1 | Recall@5 | Recall@10 |
| ----- | ----- | ----- | ----- |
| Audio → Text | TBD | TBD | TBD |
| Text → Audio | TBD | TBD | TBD |

These values should be replaced with the final Task 4 results after contrastive training and evaluation are completed.

## **Project Structure**

GNN-BERT-Music-Context/  
│  
├── README.md  
├── requirements.txt  
│  
├── notebooks/  
│   ├── 01\_setup.ipynb  
│   ├── 02\_bert.ipynb  
│   ├── 03\_gnn.ipynb  
│   ├── 04\_fusion.ipynb  
│   ├── 05\_contrastive.ipynb  
│   └── demo\_context.ipynb  
│  
├── results/  
│   ├── metrics.json  
│   ├── all\_metrics.json  
│   ├── contrastive\_metrics.json  
│   └── plots/  
│  
└── report/  
    └── final\_report.pdf

## **Notebooks**

### **`01_setup.ipynb`**

Complete development/master notebook containing the full implementation and experiments.

### **`02_bert.ipynb`**

Contains the text preprocessing, DistilBERT model, training, and evaluation for Task 1\.

### **`03_gnn.ipynb`**

Contains audio feature extraction, graph construction, GraphSAGE, and the CNN audio baseline for Task 2\.

### **`04_fusion.ipynb`**

Contains the GNN-BERT fusion model, evaluation, confusion matrix, t-SNE visualization, and case studies for Task 3\.

### **`05_contrastive.ipynb`**

Contains the dual-encoder contrastive learning system, InfoNCE training, and cross-modal retrieval evaluation for Task 4\.

### **`demo_context.ipynb`**

Provides a concise demonstration of the trained system using a test music track.

## **Limitations**

This implementation is a simplified version of the proposed GNN-BERT music-context system.

In particular:

1. The experiments use FMA-small rather than the larger datasets specified in some parts of the original task formulation.  
2. The text representation is derived from FMA metadata rather than natural-language MusicCaps captions.  
3. The classification task uses eight FMA genre classes rather than the recommended multi-label music-tag classification setup.  
4. The contrastive Task 4 implementation therefore represents a simplified MusicCaps-style audio-text retrieval experiment rather than an experiment using the full MusicCaps dataset.

These limitations should be considered when interpreting the results.

## **Requirements**

The main Python dependencies are listed in `requirements.txt`.

The project was developed using Google Colab with CPU execution.

## **Conclusion**

The project demonstrates a hybrid approach to music-context understanding using graph-based audio representations and transformer-based text representations.

The experiments compare conventional mel-spectrogram CNN features, graph-based audio representations, text-only BERT representations, and a combined GNN-BERT architecture. A contrastive learning component further explores alignment between audio and text representations through cross-modal retrieval.

