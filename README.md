# **LV Distribution Network - Topology Assessment**

<p align="center">
  <img src="config/assets/distribution_methodology.png" width="500" alt="Process flow for LV distribution network design">
</p>

---

A lightweight, interactive **Streamlit application** for planning and checking **Low-Voltage (LV) distribution networks** starting from customer locations and optional road data.

The tool is designed as a **planning and assessment aid**, not as a detailed engineering design software. Its main purpose is to help users understand how **settlement geometry**, **connection heuristics**, and **simple engineering rules** affect pole count, LV network length, and electrical feasibility.

---

## Overview

The application supports a 3-step workflow:
- **Step 1: Grid Topology**
  Build a heuristic pole-and-line layout from users and optional roads.
- **Step 2: Grid Validation**
  Aggregate demand to poles and run single-snapshot power flow checks.
- **Step 3: Grid Reinforcement (fixed topology)**
  Optimize line-capacity upgrades on existing edges only, then re-check PF.

Core outputs include:
- Interactive maps (topology, load, PF status, reinforced lines)
- Summary metrics (lengths, served vs standalone, PF violations, loading)
- Downloadable GeoJSON/CSV artifacts for GIS and post-processing

---

## Conceptual Workflow

The tool follows a clear, staged logic:

### 1. Pole placement (heuristic)

Pole locations are generated using a geometric heuristic:

- **Road-based mode**  
  If a road network is provided, candidate poles are sampled along road geometries at a user-defined spacing.

- **Free-placement mode**  
  If no roads are provided, poles are placed inside the settlement by clustering nearby unserved buildings.

During this stage, buildings are associated to poles subject to:
- a maximum allowed **user–pole distance**
- a maximum number of **users per pole**

Additional poles are iteratively created to serve buildings that remain unassociated.

---

### 2. Optional selective coverage (standalone candidates)

The tool can optionally allow **very small or isolated clusters** of buildings to remain unserved by LV:

- Clusters below a user-defined **minimum cluster size** are not assigned a pole
- These buildings are flagged as **standalone system candidates**

This feature enables more realistic modeling of sparse settlements, where extending LV infrastructure may not be economically or technically justified.

---

### 3. LV network routing (Minimum Spanning Tree)

Once all poles are placed, the tool constructs the LV network by:

- Representing poles as nodes in a weighted graph  
- Assigning edge weights equal to Euclidean cable length  
- Computing a **Minimum Spanning Tree (MST)** using Kruskal’s algorithm (via NetworkX)

This guarantees:
- A **single connected radial network**
- No loops
- Minimum total LV conductor length for the given pole locations

---

### 4. Engineering post-processing: pole-to-pole span control

To avoid unrealistically long LV spans, the tool applies an optional **post-processing step**:

- If any MST edge exceeds a user-defined **maximum pole-to-pole span**,  
  the edge is subdivided by inserting intermediate **support poles**
- This preserves:
  - a single connected network
  - total LV length
- While improving:
  - physical plausibility of individual spans
  - pole spacing realism

Importantly, this step **does not change routing decisions** — it only refines the network geometry.

---

### 5. Electrical validation (single snapshot PF)

After topology is available (from session or external files), the app can:
- Aggregate hourly demand from `building_metadata.csv` + `category_profiles.csv`
- Select slack/plant bus and electrical assumptions
- Assign line parameters (global defaults or catalog-based)
- Run a single-snapshot PF and report voltage/loading violations

---

### 6. Fixed-topology reinforcement optimization

If PF indicates overloads or voltage issues (or if the user runs it manually), the app can optimize reinforcement by:
- Keeping topology, demand distribution, and slack location fixed
- Allowing line thermal capacity expansion on existing edges only
- Minimizing upgrade cost with a simple cost-per-(km*kVA) model
- Running a post-optimization PF re-check and reporting pre/post KPIs

---

## Key Parameters (User Controls)

The main parameters exposed in the interface are:

### Pole placement & association
- **Road pole spacing [m]**  
  Distance between candidate poles sampled along roads (road mode only)

- **Max user–pole connection radius [m]**  
  Maximum distance allowed between a building and the pole serving it

- **Max users per pole [#]**  
  Upper limit on how many buildings can be connected to the same pole

### Coverage behaviour
- **Allow isolated buildings to remain unserved**  
  Enables selective coverage

- **Minimum cluster size for LV [# buildings]**  
  Minimum number of nearby buildings required to justify LV extension

### Engineering refinement
- **Max LV span between poles [m]**  
  Maximum allowed pole-to-pole span; longer segments are subdivided with support poles

---

## Outputs

### Interactive map

The app displays an interactive map with clear visual encoding:

- **Green points** – buildings served by LV  
- **Red points** – standalone system candidates (if enabled)  
- **Black points** – poles (including support poles)  
- **Blue lines** – LV network (Minimum Spanning Tree)

### Summary metrics

Key indicators are reported, including:
- Total LV network length
- Number of buildings (total / grid-served / standalone)
- Number of poles
- PF voltage/loading violations
- Reinforcement added capacity and estimated cost (Step 3)

### Downloadable GIS outputs

The following GeoJSON files can be downloaded:
- **LV poles (nodes)**  
- **LV network (edges)**  

These outputs can be directly loaded into GIS software (QGIS, ArcGIS) or used in downstream analysis.

---

## Limitations and Scope

- Distances are Euclidean (not true cable routing along roads)
- Topology stage is heuristic and does not perform detailed engineering design.
- PF stage is single-snapshot and intended for screening, not full operational studies.
- Reinforcement stage currently focuses on thermal capacity expansion on fixed edges; voltage issues may still persist in some cases.

Despite these simplifications, the tool provides **transparent, explainable, and reproducible** insights into LV network topology decisions.

---

# **Installation**

### **Conda (recommended)**

```bash
conda env create -f environment.yml
conda activate mgpy_distribution

```

### **Running**

```bash
streamlit run app.py
```

Then open:

```
http://localhost:8501
```

---

# **Contact**

**Alessandro Onori**  
📧 alessandro.onori@polimi.it  

Based on original work by **Edoardo Silvestri**

Technical Advisors  
- Riccardo Mereu — Politecnico di Milano  
- Emanuela Colombo — Politecnico di Milano

---

# **License**

European Union Public Licence (EUPL v1.1).
