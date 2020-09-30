# Problem Statement
How can we increase consumer understanding of product documentation, service agreements, and nutrition facts to result in less support interactions and a lower churn rate saving on business resources and consumer confusion?
_______________________________________________________________________________
### SMART Problem Checklist
**S**: <font color='teal'>Specific <br></font>
- Consumer understanding of product documentation.

**M**: <font color='teal'>Measurable <br></font>
- Less support interactions and lower churn rate.

**A**: <font color='teal'>Action-oriented <br></font>
- Increase consumer understanding.

**R**: <font color='teal'>Relevant <br></font>
- Product documentation, service agreements, and nutrition facts.

**T**: <font color='teal'>Time-bound <br></font>
- Saving business resources.

_______________________________________________________________________________

### Context
Users often overlook product documentation, service agreements, and nutrition facts before using a product. This can lead to confusion and frustration on both sides. Furthermore, not all documentation is created equal and strict legal criteria is often more confusing to understand than it is worth reading.

This issue applies to the following types of documentation:
- Food products
- Pharmaceuticals
- Technical documentation
- Legal contracts
_______________________________________________________________________________

### Criteria for Success
A proof of concept has been created for a specific document type where a NLP model trains on a document set, takes user input in a user-friendly fashion, and returns an accurate binary output.
_______________________________________________________________________________
### Scope of Solution Space
This will require user polling data about how often they read documentation. The project success requirements results in a proof of concept there will be no A/B testing at this time. Documentation of research scope will be required to train the model, or a pre-trained model must be used. User sentiment analysis is necessary for user input to have translated meaning. A simple and easy to use user interface must be developed to take and print user questions. It is reasonable that training data will need to be web-scraped before a useful application can be considered. It would also be useful to have a small company volunteer to use the prototype to gather user data to measure the result.

_______________________________________________________________________________

### Constraints within Solution Space
NLP requires massive amounts of computational power and data. This likely will limit us to use transfer learning. The proof of concept can successfully use Wikipedia as the base-line truth for the purposes of prototyping with the understanding that a correct result is not the same as model accuracy.

_______________________________________________________________________________
### Stakeholders
* Keeping this project within the prototyping phase means there are no company stakeholders until the project is chosen to go into the next phase. However, if a company volunteers to use the prototype, they could be the that stakeholder when going into the production phase of the project.
_______________________________________________________________________________

### Key Data Sources
- [Wikipedia API](https://en.wikipedia.org/wiki/API)
- [Nutrition facts for Starbucks Menu](https://www.kaggle.com/starbucks/starbucks-menu) ( Example company )
- Other companies may require the use of web scraping.
_______________________________________________________________________________

### Overview
*This project will be done for Springboard's Data Science Career Track as capstone three. This project pitch has been derived from the idea of a "Fact Checker" chatbot that uses NLP. I found that a automated fact-checker would be more useful to cipher through difficult to read documentation in a short amount of time. The scope of product ingredients was inspired by my fiancé's makeup business where her customers are often confused about product ingredients. If the product's ingredients and legal documentation was translated into a user-friendly chat interface, it would save her and the customers a lot of time.*

GitHub repo: https://github.com/KalenWillits/CapstoneTwo.git

Author: Kalen Willits
