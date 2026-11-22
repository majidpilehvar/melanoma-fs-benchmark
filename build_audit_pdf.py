import os, pandas as pd, matplotlib
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
T = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
pdfmetrics.registerFont(TTFont("Sans", os.path.join(T, "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("Sans-B", os.path.join(T, "DejaVuSans-Bold.ttf")))
a = pd.read_csv("results/audit/audit_all.csv")
h = ParagraphStyle("h", fontName="Sans-B", fontSize=14, leading=18, spaceAfter=6)
p = ParagraphStyle("p", fontName="Sans", fontSize=9, leading=13, spaceAfter=6)
c = ParagraphStyle("c", fontName="Sans", fontSize=6.6, leading=8.2)
cb = ParagraphStyle("cb", parent=c, fontName="Sans-B")
def fmt(v):
    try: return f"{float(v):.4g}"
    except Exception: return str(v)
rows = [[Paragraph(x, cb) for x in ["Section", "Claim in manuscript", "Stated", "Recomputed", "Source", "Result"]]]
for _, r in a.iterrows():
    rows.append([Paragraph(str(r.section), c), Paragraph(str(r.claim), c), Paragraph(fmt(r.stated_in_paper), c),
                 Paragraph(fmt(r.recomputed), c), Paragraph(str(r.source), c),
                 Paragraph(f'<font color="{"#1B7F3A" if r.result=="PASS" else "#B00020"}">{r.result}</font>', cb)])
W = A4[0] - 3.6 * cm
t = Table(rows, colWidths=[w * W for w in [0.08, 0.32, 0.14, 0.14, 0.24, 0.08]], repeatRows=1)
st = [("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.black), ("VALIGN", (0, 0), (-1, -1), "TOP"),
      ("TOPPADDING", (0, 0), (-1, -1), 1.2), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.2)]
st += [("BACKGROUND", (0, i), (-1, i), colors.HexColor("#F3F5F8")) for i in range(2, len(rows), 2)]
t.setStyle(TableStyle(st))
n, ok = len(a), int((a.result == "PASS").sum())
SimpleDocTemplate(os.environ.get("AUDIT_PDF", "Number_audit_report.pdf"), pagesize=A4, leftMargin=1.8*cm, rightMargin=1.8*cm,
                  topMargin=1.6*cm, bottomMargin=1.6*cm, title="Number audit of the melanoma manuscript").build(
    [Paragraph("Number audit of the manuscript (revision after pre-submission review)", h),
     Paragraph(f"Every key number quoted in the revised manuscript was recomputed by audit_numbers.py, from the result "
               f"files written by the analysis code or directly from Data.csv, and compared with the printed value. "
               f"<b>{ok} of {n} checks pass.</b> Cross-validation-based checks use the paper's own folds "
               f"(5-fold, stratified by class, seed 42).", p),
     Paragraph("To verify independently: <font name='Sans-B'>MELANOMA_DATA=/path/to/Data.csv bash reproduce_all.sh</font>.", p),
     Spacer(1, 4), t])
print("audit pdf", ok, n)
