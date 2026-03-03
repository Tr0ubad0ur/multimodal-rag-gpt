# FAQ

## 1. How should I write prompts to get better answers?

Use specific, goal-oriented prompts with context:

- state what exact result you need
- add constraints (format, length, style)
- if needed, specify which files/folders should be used

Good example:

> "Summarize this document in 5 bullet points and highlight implementation risks."

## 2. What file types are supported?

Supported formats:

- text: `.txt`, `.md`
- documents: `.pdf`, `.docx`
- images: `.jpg`, `.jpeg`, `.png`, `.webp`
- videos: `.mp4`, `.mov`, `.avi`, `.mkv`

## 3. Is there a file size limit?

Yes. Maximum upload size per file is **50 MB**.

Larger files will be rejected.

## 4. Can I upload via button and drag-and-drop?

Yes, both flows are supported:

- select a file with the upload button
- drag and drop a file into the attachment area

After upload, the file is indexed and becomes searchable in answers.

## 5. How do I work with folders in the knowledge base?

Typical workflow:

- create a folder (and subfolders if needed)
- upload files
- attach files to the target folder
- ask questions with folder/file filters

This keeps context focused and improves answer quality.

## 6. Can I drag and drop an entire folder?

In most cases, uploads are file-based.

Recommended approach:

- first create the folder structure in the knowledge base
- then upload files and place them into folders

## 7. How do I restrict search to specific content?

Use one of these filters:

- by folders, for topic-level scope
- by specific files, for strict source control

Narrower scope usually gives more relevant answers.

## 8. Why is the answer too generic or off-topic?

Common reasons:

- prompt is too broad
- no relevant folder/file filters were selected
- knowledge base lacks enough source material

What helps:

- rewrite the prompt with explicit intent
- request a specific output format (steps/table/bullets)
- narrow the search scope to relevant files/folders

## 9. Where can I see query history?

When authenticated, recent questions and answers are stored in history.

This helps you:

- revisit previous results
- reuse effective prompt patterns
- remove individual history items when needed

## 10. Are my data isolated from other users?

Yes, in authenticated mode data is user-scoped.

That means:

- your files and history are tied to your account
- retrieval and answers are based on your own data context

## 11. What if a file is uploaded but not reflected in answers?

Check the following:

- file is attached to the intended folder
- correct folder/file filters are selected
- file format is supported
- file size is within limits

If all checks pass, retry with a more explicit prompt referencing that file.

## 12. Are multimodal questions supported?

Yes. You can combine a text question with an attachment (for example, an image or a document) so the model can use additional context.
