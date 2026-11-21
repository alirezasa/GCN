# تمرین ۵ — طبقه‌بندی موضوعی مقالات Cora با GCN و مقایسه با MLP

| فایل | توضیح |
|---|---|
| `hw5_gcn_cora.py` | کد کامل تمرین (اسکریپت پایتون) |
| `HW5_GCN_Cora.ipynb` | همان کد به‌صورت نوت‌بوک با خروجی‌های اجراشده (مناسب Google Colab و ارائه) |
| `HW5_Report_FA.docx` / `.pdf` | گزارش فارسی تمرین |
| `HW5_Code_Explanation_FA.docx` / `.pdf` | توضیح بخش‌به‌بخش کد و معماری مدل برای ارائه + سوالات احتمالی |
| `outputs/figures/` | همه نمودارها (زیرگراف، منحنی‌های Loss/Accuracy، ماتریس درهم‌ریختگی، t-SNE، ...) |
| `outputs/results.json`, `outputs/run_log.txt` | نتایج عددی و خروجی متنی اجرا |
| `report_src/` | اسکریپت‌های ساخت گزارش Word و شکل معماری |

## نتیجه (۱۰ اجرا با seed مختلف، ۱۰۰۰ گره آزمون)

| مدل | ورودی | پارامتر | دقت آزمون |
|---|---|---|---|
| MLP | فقط بردار کلمات | 23,063 | **58.2 ± 0.8 %** |
| GCN (دو لایه) | بردار کلمات + گراف ارجاعات | 23,063 | **82.0 ± 0.8 %** |

## اجرا

```bash
pip install torch torch_geometric networkx matplotlib scikit-learn
```

```bash
python hw5_gcn_cora.py
```

ساخت دوباره گزارش‌های Word (اختیاری، نیاز به Node.js و `npm install docx`):

```bash
python report_src/make_architecture_fig.py && node report_src/build_report.js && node report_src/build_code_explanation.js
```
