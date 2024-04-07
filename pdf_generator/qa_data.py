import PyPDF2

qa_pairs = {
    "What are the top 5 good customer service skills?": "The top 5 customer service skills are problem solving, conflict resolution, multitasking, effective communication with customers and teammates, and active listening.",
    "How to demonstrate customer service skills?": "Demonstrate customer service skills on your resume by creating a dedicated section, listing skills like problem solving, conflict resolution, multitasking, effective communication, and active listening, and quantifying achievements if possible.",
    "How do I pivot into business from customer service?": "To transition into business from customer service, consider studying a relevant degree, building a network, and gaining experience in areas like sales, marketing, management, or entrepreneurship."
}

def create_pdf(qa_pairs, filename='QA.pdf'):
    pdf_writer = PyPDF2.PdfWriter()

    for question, answer in qa_pairs.items():
        pdf_writer.add_page()
        page = pdf_writer.pages[-1]
        page.mergePage(PyPDF2.PdfFileReader('pdf_generator/templates/blank_page.pdf').getPage(0)) # Use a blank page template

        content = f"Q: {question}\nA: {answer}"
        page.mergePage(PyPDF2.pdf.PageObject.create_text_object(content))

    with open(filename, 'wb') as pdf_file:
        pdf_writer.write(pdf_file)

# Call the function with your questions and answers
create_pdf(qa_pairs)
