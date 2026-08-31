import io
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

def generate_pdf_report(disease: str, confidence: float, weather: dict, advisory_text: str, gradcam_bytes: bytes) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )
    styles = getSampleStyleSheet()
    story = []

    # Title
    title_style = ParagraphStyle(
        'TitleStyle',
        parent=styles['Heading1'],
        fontSize=20,
        textColor=colors.HexColor("#1b4332")
    )
    story.append(Paragraph("Smart Crop Health Diagnostic Report", title_style))
    story.append(Spacer(1, 12))

    # Metrics Summary Table
    data = [
        ["Diagnosed Condition:", disease],
        ["Model Confidence:", f"{confidence}%"],
        ["Location Temperature:", f"{weather.get('temperature', 'N/A')}°C"],
        ["Wind Velocity:", f"{weather.get('windspeed', 'N/A')} km/h"],
        ["Disease Spread Risk:", weather.get('risk_score', 'N/A')]
    ]
    t = Table(data, colWidths=[150, 350])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#f0fdf4")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#bbf7d0")),
        ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'),
        ('PADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t)
    story.append(Spacer(1, 15))

    # Grad-CAM Heatmap Image
    if gradcam_bytes:
        story.append(Paragraph("<b>Explainable AI (Grad-CAM) Visual Lesion Mapping:</b>", styles['Normal']))
        story.append(Spacer(1, 6))
        cam_img = RLImage(io.BytesIO(gradcam_bytes), width=180, height=180)
        story.append(cam_img)
        story.append(Spacer(1, 15))

    # Advisory Section
    story.append(Paragraph("<b>Agronomic Treatment & Weather-Adapted Advisory:</b>", styles['Normal']))
    story.append(Spacer(1, 6))
    story.append(Paragraph(advisory_text, styles['BodyText']))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()