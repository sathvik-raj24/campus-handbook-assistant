# Validation on 26 October 2026

* 11 automated tests passed: chunk boundaries and metadata, zero-vector similarity, retrieval, prompt provenance, upload decoding, local HTTP index/ask flow, validation errors, failed rebuild state, and simulated embedding/generation contracts.
* Real browser check passed: built six sample chunks and displayed library evidence for the borrowing question.
* Keyword retrieval at size 100, overlap 20, k=3 retrieved the expected source and all selected evidence phrases for all eight supported questions. This is a tiny synthetic set and does not establish general retrieval or answer accuracy.
* Real local Qwen 2.5 1.5B inference returned four books and fourteen days for the borrowing question. With the final explicit citation instruction it included [S1], whose passage supports those facts.
* Real local inference on the unsupported hostel-fee question returned “I don't know from these documents.”
* The complete tested RAG path uses TF-IDF retrieval and the local generator. Semantic embeddings use the documented Ollama API and have simulated contract tests, but the Nomic embedding model was unavailable in this environment, so real semantic retrieval was not tested.
* The initial generator response omitted a source label; explicit prompting improved that sampled result. Citation and abstention behavior remain probabilistic and require manual checks.

The Ollama service was reachable at localhost even though the CLI was not on this shell's PATH. The project does not install software or download models automatically. Rehearse installation, PDF extraction, semantic retrieval, and speed on the actual workshop machines.

## Instructor handbook additions

* Reference exercise solutions checked for chunk overlap, tail preservation, empty input, invalid overlap, cosine result, zero vectors, dimension mismatch, and source numbering.
* Build-along load, chunks, search, and prompt stages passed baseline checks. Default data produced three documents, six chunks, and three retrieved hits.
* Build-along answer stage tested with real local Qwen generation: four books, fourteen days, and [S1].
* Handbook includes complete source copied from the accompanying Python files. Its internal navigation and HTML structure were checked programmatically. Browser visual inspection of the local handbook was unavailable because the browser blocks file URLs.
