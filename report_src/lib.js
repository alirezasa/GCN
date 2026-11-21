// ابزار کمکی برای ساخت سند Word راست‌به‌چپ (فارسی) با کتابخانه docx
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell,
  AlignmentType, HeadingLevel, WidthType, ShadingType, BorderStyle, LevelFormat,
  Footer, PageNumber, PageBreak, TableOfContents,
} = require("docx");

const FA_FONT = "Tahoma";
const CODE_FONT = "Consolas";
const BODY = 21; // 10.5pt

const font = (f) => ({ ascii: f, hAnsi: f, cs: f, eastAsia: f });
const LATIN = /([A-Za-z0-9_\-+*/=^().,:;%'"<>\[\]{}|\\~!?@#$&×→Â·±]+(?:[ \t]+[A-Za-z0-9_\-+*/=^().,:;%'"<>\[\]{}|\\~!?@#$&×→Â·±]+)*)/;

// تبدیل متن با نشانه‌گذاری ساده (**پررنگ** و `کد`) به TextRunها
function runs(text, base = {}) {
  const out = [];
  const tokens = text.split(/(\*\*[^*]+\*\*|`[^`]+`)/).filter((t) => t !== "");
  for (const tok of tokens) {
    let t = tok, bold = !!base.bold, code = false;
    if (t.startsWith("**")) { t = t.slice(2, -2); bold = true; }
    else if (t.startsWith("`")) { t = t.slice(1, -1); code = true; }
    if (code) {
      out.push(new TextRun({ text: t, font: font(CODE_FONT), size: base.size ? base.size - 2 : BODY - 2,
        color: "A31515", rightToLeft: false }));
      continue;
    }
    // جدا کردن تکه‌های لاتین از فارسی برای نمایش درست جهت متن
    for (const seg of t.split(LATIN).filter((s) => s !== "")) {
      const isLatin = LATIN.test(seg) && !/[؀-ۿ]/.test(seg);
      // تکه‌های لاتین که فاصله یا علامت دارند (مثل 81.9 ± 0.8) داخل LRE…PDF قرار می‌گیرند تا ترتیبشان به هم نریزد
      const txt = isLatin && /[^A-Za-z0-9_]/.test(seg.trim()) ? "\u202A" + seg + "\u202C" : seg;
      out.push(new TextRun({
        text: txt, bold, boldComplexScript: bold, italics: !!base.italics,
        font: font(FA_FONT), size: base.size || BODY, sizeComplexScript: base.size || BODY,
        color: base.color, rightToLeft: !isLatin,
      }));
    }
  }
  return out;
}

const P = (text, o = {}) => new Paragraph({
  bidirectional: true, alignment: o.align || AlignmentType.JUSTIFIED,
  spacing: { after: o.after ?? 120, line: 340 }, children: runs(text, o),
});

const H = (level, text) => new Paragraph({
  heading: [HeadingLevel.HEADING_1, HeadingLevel.HEADING_2, HeadingLevel.HEADING_3][level - 1],
  bidirectional: true, alignment: AlignmentType.RIGHT,
  children: runs(text, { bold: true, size: [32, 26, 23][level - 1], color: ["1F3864", "2E5597", "2E5597"][level - 1] }),
});

const Bullet = (text, lvl = 0) => new Paragraph({
  bidirectional: true, alignment: AlignmentType.JUSTIFIED, numbering: { reference: "bul", level: lvl },
  spacing: { after: 60, line: 320 }, children: runs(text),
});
const Num = (text, ref = "num") => new Paragraph({
  bidirectional: true, alignment: AlignmentType.JUSTIFIED, numbering: { reference: ref, level: 0 },
  spacing: { after: 60, line: 320 }, children: runs(text),
});

// بلوک کد (چپ‌به‌راست، فونت ثابت، پس‌زمینه خاکستری)
function Code(src) {
  const lines = src.replace(/\s+$/, "").split("\n");
  return lines.map((ln, i) => new Paragraph({
    bidirectional: false, alignment: AlignmentType.LEFT,
    shading: { type: ShadingType.CLEAR, fill: "F3F3F3", color: "auto" },
    border: {
      left: { style: BorderStyle.SINGLE, size: 12, color: "2E5597", space: 4 },
      ...(i === 0 ? { top: { style: BorderStyle.SINGLE, size: 2, color: "DDDDDD", space: 2 } } : {}),
      ...(i === lines.length - 1 ? { bottom: { style: BorderStyle.SINGLE, size: 2, color: "DDDDDD", space: 2 } } : {}),
    },
    spacing: { after: i === lines.length - 1 ? 160 : 0, line: 250 },
    children: [new TextRun({ text: ln.length ? ln : " ", font: font(CODE_FONT), size: 17, rightToLeft: /[؀-ۿ]/.test(ln) && !/[A-Za-z]/.test(ln) })],
  }));
}

// فرمول وسط‌چین (چپ‌به‌راست)
const Eq = (text) => new Paragraph({
  alignment: AlignmentType.CENTER, spacing: { before: 80, after: 160 },
  children: [new TextRun({ text, font: font("Cambria Math"), size: 24, italics: true })],
});

function Img(file, widthCm, caption) {
  const buf = fs.readFileSync(file);
  // خواندن ابعاد PNG از هدر
  const w = buf.readUInt32BE(16), h = buf.readUInt32BE(20);
  const wpx = widthCm * 37.8, hpx = wpx * h / w;
  const out = [new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 60 },
    children: [new ImageRun({ type: "png", data: buf, transformation: { width: wpx, height: hpx } })] })];
  if (caption) out.push(P(caption, { align: AlignmentType.CENTER, size: 18, italics: true, color: "444444", after: 200 }));
  return out;
}

// جدول راست‌به‌چپ
function Tbl(rows, widthsCm, o = {}) {
  const dxa = widthsCm.map((c) => Math.round(c * 567));
  const total = dxa.reduce((a, b) => a + b, 0);
  const border = { style: BorderStyle.SINGLE, size: 4, color: "999999" };
  const borders = { top: border, bottom: border, left: border, right: border };
  return new Table({
    width: { size: total, type: WidthType.DXA }, columnWidths: dxa,
    alignment: AlignmentType.CENTER,
    rows: rows.map((r, ri) => new TableRow({
      tableHeader: ri === 0,
      children: r.map((cell, ci) => new TableCell({
        width: { size: dxa[ci], type: WidthType.DXA }, borders,
        shading: ri === 0 ? { type: ShadingType.CLEAR, fill: "D9E2F3", color: "auto" }
          : (o.zebra && ri % 2 === 0 ? { type: ShadingType.CLEAR, fill: "F5F7FB", color: "auto" } : undefined),
        margins: { top: 60, bottom: 60, left: 100, right: 100 },
        children: [new Paragraph({ bidirectional: true, alignment: AlignmentType.CENTER, spacing: { after: 0 },
          children: runs(String(cell), { bold: ri === 0, size: 19 }) })],
      })),
    })),
  });
}

const Spacer = () => new Paragraph({ children: [] });
const Break = () => new Paragraph({ children: [new PageBreak()] });

function makeDoc(children, title) {
  const hstyle = (size, color, before) => ({
    run: { font: FA_FONT, size, bold: true, color },
    paragraph: { spacing: { before, after: 120 }, keepNext: true },
  });
  return new Document({
    title,
    styles: {
      default: { document: { run: { font: FA_FONT, size: BODY } },
        heading1: hstyle(32, "1F3864", 360), heading2: hstyle(26, "2E5597", 240), heading3: hstyle(23, "2E5597", 200) },
    },
    numbering: {
      config: [
        { reference: "bul", levels: [
          { level: 0, format: LevelFormat.BULLET, text: "●", alignment: AlignmentType.RIGHT,
            style: { paragraph: { indent: { right: 500, hanging: 280 } }, run: { size: 14 } } },
          { level: 1, format: LevelFormat.BULLET, text: "○", alignment: AlignmentType.RIGHT,
            style: { paragraph: { indent: { right: 1000, hanging: 280 } }, run: { size: 14 } } }] },
        ...["num", "num2", "num3", "num4", "num5"].map((ref) => ({ reference: ref, levels: [
          { level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.RIGHT,
            style: { paragraph: { indent: { right: 500, hanging: 320 } } } }] })),
      ],
    },
    sections: [{
      properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1300, bottom: 1300, left: 1250, right: 1250 } } },
      footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
        children: [new TextRun({ children: [PageNumber.CURRENT], font: font(FA_FONT), size: 18 })] })] }) },
      children,
    }],
  });
}

async function save(doc, file) {
  fs.writeFileSync(file, await Packer.toBuffer(doc));
  console.log("saved", file);
}

// استخراج بخشی از کد اصلی بین دو نشانه (برای اینکه کد داخل گزارش همیشه با فایل اصلی یکی باشد)
const SRC = fs.readFileSync(path.join(__dirname, "..", "hw5_gcn_cora.py"), "utf8");
function snippet(start, end) {
  const i = SRC.indexOf(start);
  if (i < 0) throw new Error("snippet start not found: " + start);
  const j = end ? SRC.indexOf(end, i + start.length) : SRC.length;
  if (j < 0) throw new Error("snippet end not found: " + end);
  return SRC.slice(i, j).replace(/\n+# %%[^\n]*\n?$/, "").replace(/\s+$/, "");
}

module.exports = { P, H, Bullet, Num, Code, Eq, Img, Tbl, Spacer, Break, makeDoc, save, snippet, runs,
  AlignmentType, Paragraph, TableOfContents };
