# Signature_Detector
Extracts signatures from documents &amp; cheques using AI/OCR. Detects white/non-white backgrounds, isolates signatures with OpenCV &amp; PyTesseract, and saves cropped results. Ideal for banking, legal, and workflow automation.

The Automated Signature Extraction System is a computer vision-based solution designed to detect and extract handwritten signatures from scanned documents and images, regardless of background variations. This system intelligently distinguishes between white-background documents (such as contracts, forms, and agreements) and non-white-background documents (such as cheques, bank slips, and receipts) to apply the most suitable extraction technique.

Key Features
✅ Automatic Background Detection – Classifies documents into white or non-white backgrounds for optimal signature extraction.
✅ Dual Extraction Methods – Uses text label detection for white-background documents and ROI-based detection for cheques and similar documents.
✅ Connected Components Analysis – Isolates signatures from noise, printed text, and other artifacts.
✅ Noise Filtering – Eliminates small specks and printed text while preserving genuine signatures.
✅ Flexible Output – Saves extracted signatures in a specified directory with structured filenames.

Use Cases
Document Processing Automation – Extract signatures from contracts, agreements, and forms.

Banking & Financial Services – Detect signatures on cheques, payment slips, and financial documents.

Legal & Compliance – Digitize signed documents for record-keeping and verification.

Workflow Automation – Integrate with OCR and document management systems for end-to-end processing.

Technical Approach
Background Detection – Uses pixel intensity analysis to determine if the document has a white background.

White-Background Processing –

Uses Tesseract OCR to detect "Signature," "Sign," or "Signed" labels.

Extracts the region above these labels where signatures typically appear.

Non-White-Background Processing –

Focuses on the bottom-right region (common for cheques).

Applies adaptive thresholding and connected components analysis to isolate signatures.

Signature Extraction –

Uses morphological operations to clean noise.

Identifies the largest connected component (likely the signature).

Applies padding and crops the exact signature region.

Technologies Used
OpenCV – Image processing, thresholding, and contour detection.

PyTesseract (Tesseract OCR) – Text detection for signature label recognition.

NumPy – Matrix operations for image manipulation.

Python – Core scripting and automation.

Expected Outcomes
High-Accuracy Signature Extraction – Works on diverse document types.

Reduced Manual Effort – Automates a traditionally manual process.

Scalable Solution – Can be integrated into larger document processing pipelines.

Future Enhancements
Deep Learning Integration – Train a CNN for better signature detection.

Multi-Signature Detection – Handle documents with multiple signatures.

Signature Verification – Compare extracted signatures against a database for authentication.

Conclusion
This system provides a robust, automated solution for extracting signatures from various document types, improving efficiency in industries like banking, legal, and corporate documentation. By combining traditional computer vision with intelligent background detection, it ensures reliable performance across different scenarios.

🚀 Potential Applications:
✔ Banking & Finance (Cheque processing, KYC documents)
✔ Legal & Compliance (Contract management)
✔ Corporate Workflows (HR forms, approvals)
✔ Government & Administration (Signed applications, permits)

🔧 Tech Stack: Python, OpenCV, PyTesseract, NumPy

📂 Output: Extracted signatures saved in a structured directory for further processing.
