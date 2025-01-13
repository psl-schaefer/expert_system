
# Notes

- My repository building around [PaperQA2](https://github.com/Future-House/paper-qa)

- Since I have all my papers in Zoter with corresponding meta data, I wanted to avoid calling the SemanticScholar and Crossref APIs

# TODO

- [ ] How to handle multiple PDFs per entry? 

    - Currently, if all PDFs have the same length, I simply use the first one

    - If the lenths are different, I just merge the PDFs

- [ ] Improve handling of indexing documents! I.e. I should save the index after each iteration of adding a document (or how slow is this saving?)