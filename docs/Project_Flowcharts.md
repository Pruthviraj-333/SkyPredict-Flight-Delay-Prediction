# Project Flowchart — Flight Delay Prediction System

## System Architecture Flowchart

```mermaid
flowchart TB
    subgraph DATA["📦 Data Collection"]
        A["Bureau of Transportation\nStatistics (BTS)"] --> B["Download 605K+\nFlight Records"]
        B --> C["October 2025\nRaw CSV Data"]
    end

    subgraph PREPROCESS["⚙️ Preprocessing"]
        C --> D["Remove Cancelled\n& Diverted Flights"]
        D --> E["601,570 Clean\nFlight Records"]
        E --> F["Feature Engineering\n19 Features Created"]
        F --> G["Encode Categorical\nVariables"]
        G --> H["Calculate Historical\nDelay Rates"]
    end

    subgraph TRAINING["🧠 Model Training"]
        H --> I["80/20 Train-Test\nStratified Split"]
        I --> J["Train: 481,256\nTest: 120,314"]
        J --> K["Train 10 ML\nAlgorithms"]
        K --> L["XGBoost"]
        K --> M["LightGBM"]
        K --> N["Random Forest"]
        K --> O["7 More Models"]
        L --> P["Compare Results\nSelect Best Model"]
        M --> P
        N --> P
        O --> P
        P --> Q["🏆 XGBoost Selected\n72.5% Accuracy\n0.772 ROC-AUC"]
    end

    subgraph VALIDATION["✅ Validation"]
        Q --> R["Test on 200 Unseen\nNovember 2025 Flights"]
        R --> S["65.0% Out-of-Sample\nAccuracy Confirmed"]
    end

    subgraph BACKEND["🖥️ Backend API"]
        Q --> T["FastAPI Server\nPython"]
        T --> U["Load Model\nat Startup"]
        U --> V["REST API\n11 Endpoints"]
    end

    subgraph FRONTEND["🌐 Frontend"]
        V --> W["Next.js Web App"]
        W --> X["Passenger View\nPredict Delays"]
        W --> Y["Staff Dashboard\nAnalytics & Charts"]
    end

    style DATA fill:#0d1424,stroke:#38bdf8,color:#f1f5f9
    style PREPROCESS fill:#0d1424,stroke:#38bdf8,color:#f1f5f9
    style TRAINING fill:#0d1424,stroke:#fbbf24,color:#f1f5f9
    style VALIDATION fill:#0d1424,stroke:#34d399,color:#f1f5f9
    style BACKEND fill:#0d1424,stroke:#8b5cf6,color:#f1f5f9
    style FRONTEND fill:#0d1424,stroke:#fb7185,color:#f1f5f9
```

---

## ML Pipeline Flowchart

```mermaid
flowchart LR
    A["Raw BTS\nCSV Data"] --> B["Clean Data\n-Cancelled\n-Diverted"]
    B --> C["Engineer\nFeatures"]
    C --> D["Split\n80/20"]
    D --> E["Train\nXGBoost"]
    E --> F["Evaluate\n72.5% Acc"]
    F --> G["Save\nModel .pkl"]

    style A fill:#111827,stroke:#38bdf8,color:#f1f5f9
    style B fill:#111827,stroke:#38bdf8,color:#f1f5f9
    style C fill:#111827,stroke:#fbbf24,color:#f1f5f9
    style D fill:#111827,stroke:#fbbf24,color:#f1f5f9
    style E fill:#111827,stroke:#34d399,color:#f1f5f9
    style F fill:#111827,stroke:#34d399,color:#f1f5f9
    style G fill:#111827,stroke:#8b5cf6,color:#f1f5f9
```

---

## Feature Engineering Flowchart

```mermaid
flowchart TD
    RAW["Raw Flight Record"] --> T["Temporal Features"]
    RAW --> R["Route Features"]
    RAW --> C["Categorical Features"]
    RAW --> H["Historical Features"]

    T --> T1["Month"]
    T --> T2["Day of Week"]
    T --> T3["Departure Hour"]
    T --> T4["Weekend Flag"]
    T --> T5["Time Block"]

    R --> R1["Distance"]
    R --> R2["Flight Duration"]
    R --> R3["Distance Group"]

    C --> C1["Airline Code"]
    C --> C2["Origin Airport"]
    C --> C3["Destination Airport"]

    H --> H1["Carrier Delay Rate"]
    H --> H2["Airport Delay Rate"]
    H --> H3["Route Delay Rate"]
    H --> H4["Hour Delay Rate"]
    H --> H5["Day-of-Week Delay Rate"]

    T1 & T2 & T3 & T4 & T5 & R1 & R2 & R3 & C1 & C2 & C3 & H1 & H2 & H3 & H4 & H5 --> MODEL["19 Feature Vector\n→ XGBoost Model"]

    style RAW fill:#111827,stroke:#38bdf8,color:#f1f5f9
    style T fill:#0b1120,stroke:#fbbf24,color:#f1f5f9
    style R fill:#0b1120,stroke:#34d399,color:#f1f5f9
    style C fill:#0b1120,stroke:#8b5cf6,color:#f1f5f9
    style H fill:#0b1120,stroke:#fb7185,color:#f1f5f9
    style MODEL fill:#111827,stroke:#38bdf8,color:#f1f5f9
```

---

## Prediction Flow (User Request)

```mermaid
flowchart LR
    A["User Enters\nFlight Details"] --> B["Frontend\nSends API Request"]
    B --> C["Backend\nReceives Request"]
    C --> D["Prepare\n19 Features"]
    D --> E["XGBoost\nModel Predicts"]
    E --> F{"Delay\nProbability?"}
    F -->|"> 50%"| G["⚠️ HIGH RISK\nLikely Delayed"]
    F -->|"30-50%"| H["🟡 ELEVATED\nModerate Risk"]
    F -->|"15-30%"| I["🔵 MODERATE\nLow-Medium Risk"]
    F -->|"< 15%"| J["✅ LOW RISK\nLikely On-Time"]
    G & H & I & J --> K["Display Result\nwith Gauge Chart"]

    style A fill:#111827,stroke:#38bdf8,color:#f1f5f9
    style B fill:#111827,stroke:#38bdf8,color:#f1f5f9
    style C fill:#111827,stroke:#fbbf24,color:#f1f5f9
    style D fill:#111827,stroke:#fbbf24,color:#f1f5f9
    style E fill:#111827,stroke:#34d399,color:#f1f5f9
    style F fill:#111827,stroke:#fbbf24,color:#f1f5f9
    style G fill:#1a0000,stroke:#fb7185,color:#fb7185
    style H fill:#1a1400,stroke:#fbbf24,color:#fbbf24
    style I fill:#001a1a,stroke:#38bdf8,color:#38bdf8
    style J fill:#001a00,stroke:#34d399,color:#34d399
    style K fill:#111827,stroke:#8b5cf6,color:#f1f5f9
```

---

## Model Comparison Flowchart

```mermaid
flowchart TD
    D["Training Data\n481,256 Flights"] --> A["10 ML Algorithms"]

    A --> X["XGBoost\n72.5% | AUC 0.772"]
    A --> L["LightGBM\n70.3% | AUC 0.759"]
    A --> RF["Random Forest\n71.6% | AUC 0.749"]
    A --> GB["Gradient Boosting\n81.7% | AUC 0.757"]
    A --> ET["Extra Trees\n67.9% | AUC 0.737"]
    A --> AB["AdaBoost\n79.7% | AUC 0.707"]
    A --> DT["Decision Tree\n66.2% | AUC 0.720"]
    A --> LR["Logistic Regression\n64.9% | AUC 0.708"]
    A --> KN["KNN\n80.5% | AUC 0.713"]
    A --> SV["Linear SVM\n64.8% | AUC 0.708"]

    X --> W["🏆 Winner: XGBoost\nBest ROC-AUC\nBest Balance of\nAccuracy + Recall"]

    GB -.->|"High accuracy but\n<20% delay recall"| TRAP["⚠️ Accuracy Trap\nPredicts everything\nas on-time"]
    KN -.-> TRAP
    AB -.-> TRAP

    style D fill:#111827,stroke:#38bdf8,color:#f1f5f9
    style A fill:#111827,stroke:#fbbf24,color:#f1f5f9
    style X fill:#0b1120,stroke:#34d399,color:#34d399
    style W fill:#0b1120,stroke:#34d399,color:#34d399
    style TRAP fill:#1a0000,stroke:#fb7185,color:#fb7185
    style L fill:#111827,stroke:#38bdf8,color:#f1f5f9
    style RF fill:#111827,stroke:#38bdf8,color:#f1f5f9
    style GB fill:#111827,stroke:#fb7185,color:#fb7185
    style KN fill:#111827,stroke:#fb7185,color:#fb7185
    style AB fill:#111827,stroke:#fb7185,color:#fb7185
```

---

## Dual-Model Architecture (Future Scope)

```mermaid
flowchart TD
    INPUT["User Flight\nRequest"] --> CHECK{"Weather Data\nAvailable?"}

    CHECK -->|"Yes"| PRIMARY["Primary Model\nFlight + Weather Data\n~80-85% Expected Accuracy"]
    CHECK -->|"No"| FALLBACK["Fallback Model\nFlight Data Only\n72.5% Accuracy"]

    PRIMARY --> RESULT["Prediction\nResult"]
    FALLBACK --> RESULT

    RESULT --> UI["Display to User"]

    style INPUT fill:#111827,stroke:#38bdf8,color:#f1f5f9
    style CHECK fill:#111827,stroke:#fbbf24,color:#f1f5f9
    style PRIMARY fill:#0b1120,stroke:#34d399,color:#34d399
    style FALLBACK fill:#0b1120,stroke:#38bdf8,color:#38bdf8
    style RESULT fill:#111827,stroke:#8b5cf6,color:#f1f5f9
    style UI fill:#111827,stroke:#8b5cf6,color:#f1f5f9
```

---

## Project Progress Timeline

```mermaid
gantt
    title Project Progress
    dateFormat  YYYY-MM-DD
    section Data Collection
        Download BTS Data (605K flights)    :done, d1, 2026-02-12, 1d
        Data Cleaning & Validation          :done, d2, after d1, 1d
    section Feature Engineering
        Create 19 Features                  :done, f1, after d2, 1d
        Calculate Aggregate Rates           :done, f2, after f1, 1d
    section Model Training
        Train XGBoost Model                 :done, t1, after f2, 1d
        Compare 10 Algorithms               :done, t2, after t1, 1d
        Out-of-Sample Validation            :done, t3, after t2, 1d
    section Web Application
        Build FastAPI Backend               :done, w1, after t3, 1d
        Build Next.js Frontend              :done, w2, after w1, 2d
        UI Redesign & Polish                :done, w3, after w2, 1d
    section Future Work
        Primary Model (with Weather)        :active, p1, 2026-02-24, 7d
        Deployment                          :        p2, after p1, 3d
```
