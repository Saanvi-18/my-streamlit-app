import streamlit as st
import pandas as pd
import numpy as np
import io, zipfile, json, re, base64
from datetime import datetime, date, timedelta
import matplotlib.pyplot as plt
import feedparser

# ----------------------- Synthetic Capabilities / Library -----------------------
CAPABILITIES = {
    "services": ["LLM/GenAI", "RPA", "GIS/Digital Twin", "Data Platform",
                 "Cyber/Zero Trust", "Edge GPU/5G", "Digital Health AI"],
    "frameworks": {"G-Cloud 14": True, "DOS 7": True, "NHS SBS": True,
                   "RM6263 Cyber": False, "JOSCAR": False, "DFOCVC": True},
    "certifications": {"ISO 27001": True, "Cyber Essentials Plus": True,
                       "ISO 13485": False, "DCB0129": True, "DCB0160": True,
                       "NPPV3": False, "SC": False, "DV": False},
    "case_studies": ["nhs_ai_pathology", "la_data_platform", "edge_gpu_cv"],
    "differentiators": [
        "Agentic AI workflow with guardrails and audit",
        "Reusable accelerators: RAG toolkit, Social Value planner, Compliance matrix",
        "Edge-native GPU micro-datacenter design for low-latency AI",
        "Healthcare clinical safety (DCB0129/0160) experience",
    ],
}
CASE_LIB = {
"nhs_ai_pathology": """# Case Study: NHS AI Pathology
**Context:** Assisted an NHS Trust to trial AI triage models over digitised slides with LIMS integration.
**Outcomes:** 22% faster reporting turnaround, improved QC, and scalable vendor-neutral architecture.
**Relevance:** Alignment to all-Wales digital cellular pathology with national LIMS and VR integration.
""",
"la_data_platform": """# Case Study: Local Authority Data Platform
**Context:** Delivered a lakehouse with governed self-service analytics for planning and waste services.
**Outcomes:** 18% improvement in SLA adherence; citizen query response time reduced by 35%.
**Relevance:** Matches data platform + LLM insight requirements.
""",
"edge_gpu_cv": """# Case Study: Edge GPU for Computer Vision
**Context:** Deployed edge GPU nodes with model orchestration for transport safety analytics.
**Outcomes:** 40% incident detection improvement and <100ms inference latency at the edge.
**Relevance:** Aligns with private 5G + MEC video analytics tenders.
"""
}
ISO_SUMMARY = """# Policy Summary: ISO 27001 ISMS
We maintain an ISMS aligned to ISO 27001 with documented SoA, risk treatment, and PDCA improvement.
"""
SOCIAL_VALUE = """# Social Value Plan (UK TOMs)
- Local apprenticeships and skills bootcamps
- Net-zero delivery plan and green datacenter options
- SME supply chain participation and community grants
"""
TEAM_CV = """# Team CV (Template)
**Name:** Jane Smith, AI Solutions Lead  
**Certs:** TOGAF, Azure Architect, ISO 27001 LI  
**Experience:** 12 years delivering public sector AI and data platforms.
"""

DEFAULT_WEIGHTS = {"fit_weight":0.35,"value_weight":0.25,"deadline_weight":0.15,"compliance_weight":0.15,"capacity_weight":0.10}

# ----------------------- Synthetic Tender Feed -----------------------
def synth_tenders(today=date(2025,8,26)):
    rows = [
        {"id":"UK-NHS-001","title":"NHS Wales National Digital Cellular Pathology Programme – AI Decision Support","buyer":"NHS Wales Shared Services Partnership","portal":"Atamis","deadline":(today+timedelta(days=18)).isoformat(),"est_value_gbp":5500000,"cpv_codes":"48814000;85111800;72200000","location":"Wales","framework_required":"NHS SBS;DSPT;Cyber Essentials Plus","summary":"Digital pathology scanning platform with AI triage, LIMS, VR, all-Wales access.","requirements":"AI/ML;Interoperability;LIMS integration;ISO 27001;InfoGov;On-prem + Cloud;Vendor-neutral","question_count":24,"eval_weights":"Quality:70;Price:20;SocialValue:10"},
        {"id":"UK-CCS-002","title":"Conversational AI & RPA for Citizen Services","buyer":"Cabinet Office (via CCS)","portal":"Jaggaer","deadline":(today+timedelta(days=11)).isoformat(),"est_value_gbp":1500000,"cpv_codes":"72212100;72212222;48218000","location":"England","framework_required":"G-Cloud 14","summary":"Omnichannel assistant, RPA, knowledge base, WCAG 2.2.","requirements":"LLM;RPA;Knowledge Graph;WCAG;PIA/DPIA;Monitoring;Guardrails","question_count":18,"eval_weights":"Quality:60;Price:30;SocialValue:10"},
        {"id":"UK-POL-003","title":"Police Force Secure Cloud Migration & Zero Trust","buyer":"Metropolitan Police","portal":"Bravo","deadline":(today+timedelta(days=28)).isoformat(),"est_value_gbp":4200000,"cpv_codes":"72514300;72222300;72317000","location":"London","framework_required":"RM6263 Cyber Security Services 3","summary":"Migration with zero trust, SOC, SIEM/XDR, NPPV3.","requirements":"Azure;ZeroTrust;SOC;SIEM;XDR;ISO27001;SC/NPPV3","question_count":32,"eval_weights":"Quality:60;Price:25;Security:15"},
        {"id":"EU-HE-004","title":"Horizon Europe Pilot: AI for Flood Risk Digital Twins","buyer":"EU Research Directorate","portal":"Funding & Tenders","deadline":(today+timedelta(days=36)).isoformat(),"est_value_gbp":900000,"cpv_codes":"71354100;72316000;71200000","location":"EU","framework_required":"HE Consortium","summary":"Hydrology digital twin w/ ML, edge sensors, satellite.","requirements":"GIS;DigitalTwin;Edge;ML;ESA Copernicus;OpenData;Ethics","question_count":12,"eval_weights":"Excellence:50;Impact:30;Implementation:20"},
        {"id":"UK-LA-005","title":"Local Authority Data Platform & AI Insights","buyer":"Birmingham City Council","portal":"Proactis","deadline":(today+timedelta(days=9)).isoformat(),"est_value_gbp":800000,"cpv_codes":"48810000;72316000;72222300","location":"England","framework_required":"DOS 7 or G-Cloud 14","summary":"Data lakehouse, dashboards, LLM insights.","requirements":"DataPlatform;LLM;Dashboards;PIA;DPO signoff;MLOps","question_count":16,"eval_weights":"Quality:55;Price:35;SocialValue:10"},
        {"id":"UK-NHS-006","title":"NHS Trust Virtual Ward with AI Deterioration Alerts","buyer":"NHS Trust (North West)","portal":"Atamis","deadline":(today+timedelta(days=14)).isoformat(),"est_value_gbp":1200000,"cpv_codes":"85111800;72200000;48814000","location":"England","framework_required":"NHS SBS;DFOCVC","summary":"Remote monitoring, AI risk, EPR integration, clinical safety.","requirements":"AI Risk;FHIR;EPR;DCB0129;DCB0160;CE;ISO13485 (preferred)","question_count":20,"eval_weights":"Quality:65;Price:25;ClinicalSafety:10"},
        {"id":"UK-EDU-007","title":"AI Proctored Remote Exams Platform","buyer":"Russell Group University","portal":"In-tend","deadline":(today+timedelta(days=20)).isoformat(),"est_value_gbp":600000,"cpv_codes":"80420000;72222300","location":"England","framework_required":"SUPC","summary":"Secure online exams with AI proctoring, privacy-by-design.","requirements":"AI;Privacy;WCAG;Appeals;SCCs;Data Residency","question_count":14,"eval_weights":"Quality:50;Price:40;Accessibility:10"},
        {"id":"UK-TEL-008","title":"5G Private Network with Edge GPU for Video Analytics","buyer":"Transport for London","portal":"Jaggaer","deadline":(today+timedelta(days=22)).isoformat(),"est_value_gbp":3100000,"cpv_codes":"32412100;64200000;72222300","location":"London","framework_required":"N/A","summary":"Private 5G with UPF on-prem, MEC GPU CV analytics.","requirements":"5G;UPF;MEC;GPU;Video Analytics;Data Sharing","question_count":26,"eval_weights":"Quality:55;Price:30;Innovation:15"},
        {"id":"UK-DEF-009","title":"Defence Digital – Edge Compute & AI Inference","buyer":"MOD Defence Digital","portal":"AWARD","deadline":(today+timedelta(days=30)).isoformat(),"est_value_gbp":7500000,"cpv_codes":"73436000;72200000","location":"UK-wide","framework_required":"JOSCAR;Security Clearance SC/DV","summary":"Rugged edge compute, model pipeline, air-gapped.","requirements":"Edge;Airgap;SC;SBOM;ZeroTrust;DevSecOps","question_count":30,"eval_weights":"Quality:60;Security:25;Price:15"},
        {"id":"UK-HEALTH-010","title":"Population Health Analytics & Social Value Programme","buyer":"Greater Manchester ICS","portal":"Atamis","deadline":(today+timedelta(days=17)).isoformat(),"est_value_gbp":2200000,"cpv_codes":"73110000;72316000","location":"England","framework_required":"NHS SBS;G-Cloud 14","summary":"Risk stratification, inequalities insights, SV.","requirements":"Analytics;Inequalities;LLM;Ethical AI;SV Framework","question_count":21,"eval_weights":"Quality:60;Price:25;SocialValue:15"},
        {"id":"UK-LG-011","title":"Planning Department – LLM Assistant & Document Automation","buyer":"Leeds City Council","portal":"Proactis","deadline":(today+timedelta(days=7)).isoformat(),"est_value_gbp":450000,"cpv_codes":"72320000;48211000","location":"England","framework_required":"G-Cloud 14","summary":"Drafting, summarisation, search, redaction.","requirements":"RAG;Redaction;LLM;PIA;FOI;Audit","question_count":10,"eval_weights":"Quality:55;Price:35;DataGov:10"},
        {"id":"EU-URBAN-012","title":"EU Urban Mobility – Computer Vision for Safety","buyer":"European Innovation Council","portal":"Funding & Tenders","deadline":(today+timedelta(days=45)).isoformat(),"est_value_gbp":500000,"cpv_codes":"34996000;72222300","location":"EU","framework_required":"HE Consortium","summary":"AI video analytics to reduce accidents; privacy-preserving.","requirements":"Vision;Edge;Privacy;OpenData;Ethics","question_count":12,"eval_weights":"Excellence:45;Impact:35;Implementation:20"},
    ]
    return pd.DataFrame(rows)

# ----------------------- LIVE RSS connector (single file) -----------------------
DATE_PAT = re.compile(
    r"(closing date|deadline|tenders? end|submission deadline)\s*[:\-]?\s*"
    r"(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}|\d{1,2}\s+\w+\s+\d{4})", re.I)
GBP_PAT  = re.compile(r"£\s?([\d,]+)")

def _parse_deadline_from_text(text: str, published: datetime | None) -> str:
    text = text or ""
    m = DATE_PAT.search(text)
    if m:
        raw = m.group(2)
        for fmt in ("%d/%m/%Y","%d-%m-%Y","%d %B %Y","%d %b %Y","%Y-%m-%d"):
            try:
                return datetime.strptime(raw, fmt).date().isoformat()
            except Exception:
                pass
    # Fallback: published + 21 days
    if published:
        return (published + timedelta(days=21)).date().isoformat()
    return (datetime.utcnow() + timedelta(days=21)).date().isoformat()

def _parse_value_gbp(text: str) -> int:
    if not text: return 0
    m = GBP_PAT.search(text.replace("\u00A0"," "))
    return int(m.group(1).replace(",","")) if m else 0

def fetch_rss_tenders(feed_url: str, portal_label: str = "RSS") -> pd.DataFrame:
    feed = feedparser.parse(feed_url)
    rows = []
    for e in feed.entries:
        title   = getattr(e, "title", "").strip()
        link    = getattr(e, "link", "").strip()
        summary = getattr(e, "summary", "") or getattr(e, "description", "")
        buyer   = getattr(e, "author", "") or (getattr(e, "tags", [{}])[0].get("term","") if getattr(e, "tags", None) else "")
        published = None
        if getattr(e, "published_parsed", None):
            published = datetime(*e.published_parsed[:6])
        rows.append({
            "id": getattr(e, "id", link) or link or title,
            "title": title or "Untitled Notice",
            "buyer": buyer or "Unknown",
            "portal": portal_label,
            "deadline": _parse_deadline_from_text(summary, published),
            "est_value_gbp": _parse_value_gbp(summary),
            "cpv_codes": "",
            "location": "UK/EU",
            "framework_required": "",
            "summary": summary,
            "requirements": "",
            "question_count": 0,
            "eval_weights": "",
        })
    return pd.DataFrame(rows)

# ----------------------- Helpers -----------------------
def parse_req_list(s):
    if pd.isna(s): return []
    return [x.strip() for x in str(s).replace(",", ";").split(";") if x.strip()]

def capability_fit(tender, caps):
    reqs = set(parse_req_list(tender.get("requirements","")))
    services = set([s.lower() for s in caps["services"]])
    mapping = {
        "llm":"llm/genai","rpa":"rpa","gis":"gis/digital twin","digitaltwin":"gis/digital twin",
        "edge":"edge gpu/5g","gpu":"edge gpu/5g","5g":"edge gpu/5g",
        "ai":"llm/genai","ml":"llm/genai","vision":"edge gpu/5g","data platform":"data platform",
        "zero":"cyber/zero trust","soc":"cyber/zero trust","siem":"cyber/zero trust",
        "epr":"digital health ai","lims":"digital health ai","clinical":"digital health ai"
    }
    covered = 0; notes=[]
    for r in reqs:
        key = r.lower(); hit=None
        for k,v in mapping.items():
            if k in key:
                hit = v; break
        if hit and any(hit in s for s in services):
            covered += 1; notes.append(f"Service match: {r}")
    svc_score = min(1.0, covered/max(1,len(reqs))) if reqs else 0.5
    needed = parse_req_list(tender.get("framework_required",""))
    cert_map = caps.get("certifications",{}); fw_map = caps.get("frameworks",{})
    compliance_hits=0
    for need in needed:
        if (need in fw_map and fw_map[need]) or (need in cert_map and cert_map[need]):
            compliance_hits += 1; notes.append(f"Prereq ok: {need}")
    comp_score = (compliance_hits/max(1,len(needed))) if needed else 1.0
    fit = 0.7*svc_score + 0.3*comp_score
    return float(fit), notes

def compliance_gap(tender, caps):
    needed = parse_req_list(tender.get("framework_required",""))
    cert_map = caps.get("certifications",{}); fw_map = caps.get("frameworks",{})
    missing=[]
    for need in needed:
        if (need in fw_map and not fw_map[need]) or (need in cert_map and not cert_map[need]):
            missing.append(need)
    gap = 1.0 - (len(needed)-len(missing))/max(1,len(needed)) if needed else 0.0
    return float(gap), missing

def deadline_urgency(deadline_str):
    try:
        d = datetime.fromisoformat(str(deadline_str)).date()
    except Exception:
        return 0.5
    days = (d - date.today()).days
    if days <= 0: return 0.0
    if days < 7:  return 1.0
    if days < 14: return 0.8
    if days < 21: return 0.6
    if days < 30: return 0.4
    return 0.3

def capacity_ok(): return 0.8

def score_tender(t, caps, weights):
    fit, fit_notes = capability_fit(t, caps)
    comp_gap, missing = compliance_gap(t, caps)
    urgency = deadline_urgency(t["deadline"])
    value = float(t.get("est_value_gbp", 0) or 0)
    val_score = min(1.0, np.log1p(value)/np.log1p(8_000_000))
    capacity = capacity_ok()
    s = (weights["fit_weight"]*fit + weights["value_weight"]*val_score +
         weights["deadline_weight"]*urgency + weights["compliance_weight"]*(1.0-comp_gap) +
         weights["capacity_weight"]*capacity)
    breakdown = {"fit":fit,"value":val_score,"deadline":urgency,"compliance":1.0-comp_gap,"capacity":capacity}
    notes = fit_notes + ([f"Missing: {m}" for m in missing] if missing else [])
    return float(s), breakdown, notes

def make_proposal_md(t, caps):
    cases = "\n\n".join(CASE_LIB[name] for name in caps.get("case_studies",[]) if name in CASE_LIB)
    md = f"""# Proposal: {t['title']}
**Buyer:** {t['buyer']}  
**Tender ID:** {t['id']}  
**Portal:** {t['portal']}  
**Deadline:** {t['deadline']}  
**Estimated Value:** £{int(float(t.get('est_value_gbp',0))):,}

## Executive Summary
We propose to deliver the requested solution using our Agentic AI workflow – enabling compliant, explainable automation across discovery, delivery, and continuous improvement. Our approach emphasises guardrails, audit, and measurable outcomes.

## Approach & Methodology
- Requirements coverage: {t.get('requirements','')}
- Standards & compliance: {t.get('framework_required','')}
- Delivery: Discovery → Alpha → Beta → Live, with monthly value drops
- Governance: DPO, Clinical Safety (where applicable), Security by Design

## Technical Solution
- Data & Integration: Interoperable APIs; vendor-neutral architecture
- AI: RAG with policy guardrails and audit; role-aligned agents
- Ops: MLOps/ModelOps with human-in-the-loop risk controls

## Social Value
{SOCIAL_VALUE}

## Information Security
{ISO_SUMMARY}

## Relevant Experience
{cases}

## Team
{TEAM_CV}

## Pricing (Illustrative / To Be Confirmed)
- Discovery & Mobilisation: £45,000
- Build & Configuration: £{max(60_000, int(0.05 * float(t.get('est_value_gbp',0) or 0))):,}
- Run (12 months): £{max(120_000, int(0.15 * float(t.get('est_value_gbp',0) or 0))):,}

*Final pricing and milestones will be refined during clarification.*

## Assumptions & Dependencies
- Timely access to buyer stakeholders and systems
- Availability of test data and environments
- Framework/compliance prerequisites satisfied

## Risks & Mitigations
- Model drift → MLOps and continuous evaluation
- Data quality → Data contracts and validation
- Change management → Training and stakeholder onboarding

## Conclusion
We will deliver a compliant, outcome-focused solution that de-risks delivery and maximises public value.
"""
    return md

def build_submission_zip(t, proposal_md):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(f"{t['id']}/proposal.md", proposal_md.encode("utf-8"))
        reqs = parse_req_list(t.get("requirements",""))
        comp_csv = "Requirement,Response\n" + "\n".join([f"{r},Covered" for r in reqs]) + ("\n" if reqs else "")
        z.writestr(f"{t['id']}/compliance_matrix.csv", comp_csv.encode("utf-8"))
        pricing_rows = [
            ("Discovery", 45000),
            ("Build", max(60000, int(0.05 * float(t.get("est_value_gbp",0) or 0)))),
            ("Run-12m", max(120000, int(0.15 * float(t.get("est_value_gbp",0) or 0))))
        ]
        pricing_csv = "Item,PriceGBP\n" + "\n".join(f"{i},{p}" for i,p in pricing_rows) + "\n"
        z.writestr(f"{t['id']}/pricing.csv", pricing_csv.encode("utf-8"))
    buf.seek(0)
    return buf

# ----------------------- UI -----------------------
st.set_page_config(page_title="Agentic AI RFP Demo", layout="wide")
st.title("Agentic AI Workflow for Public Sector RFPs")
st.caption("Crawl → Classify → Score → Auto-Propose → Submit → Learn")

if "weights" not in st.session_state: st.session_state.weights = DEFAULT_WEIGHTS.copy()
if "history" not in st.session_state: st.session_state.history = pd.DataFrame(
    columns=["id","title","buyer","score","result","notes","date"]
)

# Source selector: live RSS or synthetic
with st.sidebar:
    st.header("Data Source")
    use_live = st.toggle("Use LIVE RSS/Atom feed", value=False, help="Paste a watchlist/feed URL from Contracts Finder, Proactis, In-tend, etc.")
    feed_url = st.text_input("Feed URL", placeholder="https://... your RSS/Atom ...")
    portal_label = st.text_input("Portal label", value="RSS")

# Load tenders
if use_live and feed_url.strip():
    try:
        tenders = fetch_rss_tenders(feed_url.strip(), portal_label.strip() or "RSS")
        if tenders.empty:
            st.warning("Live feed returned no items. Falling back to synthetic data.")
            tenders = synth_tenders()
    except Exception as ex:
        st.error(f"Live feed error: {ex}")
        tenders = synth_tenders()
else:
    tenders = synth_tenders()

tabs = st.tabs(["1) Crawl","2) Classify","3) Score","4) Auto-Propose","5) Submit","6) Learn"])

# ----------------------- 1) Crawl -----------------------
with tabs[0]:
    st.subheader("Tender Feed")
    col1,col2,col3,col4 = st.columns(4)
    with col1: portal = st.selectbox("Portal filter", ["All"] + sorted(tenders["portal"].unique().tolist()))
    with col2: region = st.selectbox("Region filter", ["All"] + sorted(tenders["location"].unique().tolist()))
    with col3: max_days = st.slider("Deadline within (days)", 7, 60, 45, 1)
    with col4: min_value = st.number_input("Min est. value (£)", value=300000, step=50000)

    df = tenders.copy()
    if portal!="All": df=df[df["portal"]==portal]
    if region!="All": df=df[df["location"]==region]

    # ✅ Correct datetime handling (no .dt on Python dates)
    deadlines = pd.to_datetime(df["deadline"], errors="coerce", dayfirst=True)
    today_norm = pd.Timestamp.today().normalize()
    days_left = (deadlines - today_norm).dt.days
    mask = deadlines.notna() & days_left.between(0, max_days) & (df["est_value_gbp"] >= min_value)
    df = df[mask].copy()

    st.dataframe(df[["id","title","buyer","portal","deadline","est_value_gbp","framework_required"]].sort_values("deadline"))

# ----------------------- 2) Classify -----------------------
with tabs[1]:
    st.subheader("Classification & Compliance")
    if tenders.empty:
        st.info("No tenders to classify.")
    else:
        selected = st.selectbox("Select a tender", tenders["id"].tolist())
        t = tenders[tenders["id"]==selected].iloc[0].to_dict()
        fit, notes = capability_fit(t, CAPABILITIES)
        gap, missing = compliance_gap(t, CAPABILITIES)
        st.write(f"**Capability Fit:** {fit:.2f}  |  **Compliance Gap:** {gap:.2f}")
        st.write("**Notes:**")
        for n in notes[:10]: st.write("- ", n)
        if missing: st.warning("Missing prerequisites: " + ", ".join(missing))
        else: st.success("No missing prerequisites detected.")

# ----------------------- 3) Score -----------------------
with tabs[2]:
    st.subheader("Scoring & Prioritisation")
    rows=[]
    for _, r in tenders.iterrows():
        s, breakdown, _ = score_tender(r.to_dict(), CAPABILITIES, st.session_state.weights)
        rows.append({"id":r["id"],"title":r["title"],"score":s,**breakdown})
    score_df = pd.DataFrame(rows).sort_values("score", ascending=False)
    st.dataframe(score_df[["id","title","score","fit","value","deadline","compliance","capacity"]])

    top5 = score_df.head(5)
    fig, ax = plt.subplots()
    ax.barh(top5["id"], top5["score"])
    ax.set_xlabel("Score"); ax.set_ylabel("Tender ID"); ax.set_title("Top 5 Opportunities")
    st.pyplot(fig)

# ----------------------- 4) Auto-Propose -----------------------
with tabs[3]:
    st.subheader("Auto-Drafted Proposal")
    if tenders.empty:
        st.info("No tenders available.")
    else:
        selected = st.selectbox("Select tender to draft", tenders["id"].tolist(), key="draft")
        t = tenders[tenders["id"]==selected].iloc[0].to_dict()
        proposal_md = make_proposal_md(t, CAPABILITIES)
        st.download_button("Download Draft (MD)", proposal_md, file_name=f"{t['id']}_proposal.md")
        st.markdown(proposal_md)

# ----------------------- 5) Submit -----------------------
with tabs[4]:
    st.subheader("Submission Pack")
    if tenders.empty:
        st.info("No tenders available.")
    else:
        selected = st.selectbox("Select tender to package", tenders["id"].tolist(), key="submit")
        t = tenders[tenders["id"]==selected].iloc[0].to_dict()
        proposal_md = make_proposal_md(t, CAPABILITIES)
        zbuf = build_submission_zip(t, proposal_md)
        st.download_button("Download Submission ZIP", zbuf, file_name=f"{t['id']}_submission.zip")
        st.info("Simulated portal upload (Jaggaer/Atamis/etc.) with an audit-ready pack.")

# ----------------------- 6) Learn -----------------------
with tabs[5]:
    st.subheader("Learning Loop")
    if tenders.empty:
        st.info("No tenders available.")
    else:
        sel = st.selectbox("Select tender outcome to record", tenders["id"].tolist(), key="learn")
        srow = tenders[tenders["id"]==sel].iloc[0]
        s, breakdown, _ = score_tender(srow.to_dict(), CAPABILITIES, st.session_state.weights)
        result = st.radio("Outcome", ["Win","Loss","Pending"], index=2)
        notes_in = st.text_area("Notes (buyer feedback, evaluator scores)")
        if st.button("Record Outcome"):
            new = pd.DataFrame([{
                "id": srow["id"], "title": srow["title"], "buyer": srow["buyer"],
                "score": s, "result": result, "notes": notes_in, "date": date.today().isoformat()
            }])
            st.session_state.history = pd.concat([st.session_state.history, new], ignore_index=True)
            st.success("Outcome recorded.")
        st.markdown("### Win/Loss History")
        st.dataframe(st.session_state.history.tail(10))

        st.markdown("### Adjust Scoring Weights")
        w = st.session_state.weights
        fit_w = st.slider("Fit weight", 0.0, 1.0, float(w["fit_weight"]), 0.05)
        value_w = st.slider("Value weight", 0.0, 1.0, float(w["value_weight"]), 0.05)
        deadline_w = st.slider("Deadline weight", 0.0, 1.0, float(w["deadline_weight"]), 0.05)
        comp_w = st.slider("Compliance weight", 0.0, 1.0, float(w["compliance_weight"]), 0.05)
        cap_w = st.slider("Capacity weight", 0.0, 1.0, float(w["capacity_weight"]), 0.05)
        total = fit_w + value_w + deadline_w + comp_w + cap_w
        if total == 0: total = 1
        st.session_state.weights = {
            "fit_weight": fit_w/total, "value_weight": value_w/total,
            "deadline_weight": deadline_w/total, "compliance_weight": comp_w/total,
            "capacity_weight": cap_w/total
        }
        st.code(json.dumps(st.session_state.weights, indent=2))

# Sidebar info
st.sidebar.header("Roles, Data Sources & Tools")
st.sidebar.markdown("""
**Roles**  
• Crawl Agent → portals/APIs → tender metadata  
• Classifier Agent → capability & compliance labelling  
• Scoring Agent → bid/no-bid recommendation  
• Proposal Agent → RAG draft + compliance matrix  
• Submission Agent → RPA portal upload (simulated)  
• Learning Agent → win/loss analytics & model tuning  

**Data Sources**  
• LIVE (optional): Paste an RSS/Atom watchlist URL (Contracts Finder / Proactis / In-tend / Delta)  
• Synthetic: Built-in sample tenders

**Tools**  
• Streamlit UI, Pandas/Numpy, Matplotlib charts  
• Rule-based classifier & heuristic scorer (demo)
""")
