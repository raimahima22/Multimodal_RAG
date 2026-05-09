# Generation Pipeline

## 1. Purpose

The generation pipeline uses retrieved document pages to produce grounded multimodal answers.

---

## 2. Context Construction

Retrieved pages are:

- loaded as images
- converted to base64
- passed directly to the vision-language model

---

## 3. Prompting Strategy

The prompt enforces:

- factual answering
- concise responses
- document grounding
- hallucination reduction

---

## 4. Why Vision Input Is Used

Images are sent directly to the model instead of relying only on OCR text.

This preserves:

- tables
- layouts
- visual semantics
- charts
- formatting structure