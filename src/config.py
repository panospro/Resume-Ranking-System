EDUCATION_LEVELS = {
    'PhD': r'\b(Ph\.?D\.?|Doctor of Philosophy)\b',
    'Masters': r'\b(M\.?Sc\.?|M\.?Tech\.?|M\.?E\.?|M\.?Eng\.?|MBA|Master[’\'s]* of [A-Za-z]+)\b',
    'Bachelors': r'\b(B\.?Sc\.?|B\.?Tech\.?|B\.?E\.?|B\.?Eng\.?|B\.?Com\.?|B\.?C\.?A\.?|B\.?B\.?A\.?|Bachelor[’\'s]* of [A-Za-z]+)\b',
    'Diploma': r'\b(Diploma|Polytechnic|Associate Degree)\b',
    'High School': r'\b(High School|12th|10th|Secondary|Senior Secondary|HSC|SSC)\b'
}

EDU_RANK = {
    "PhD": 5,
    "Masters": 4,
    "MBA": 4,
    "Bachelor's": 3,
    "Bachelors": 3,
    "Possibly Bachelors": 3,
    "Diploma": 2,
    "High School": 1,
    "Unknown": 0,
    "Not Specified": 0
}

TECH_SKILLS = [
    # Programming Languages
    "python", "java", "c++", "c#", "javascript", "typescript", "go", "rust", "ruby", "php", "swift", "kotlin", "r",

    # Databases
    "sql", "mysql", "postgresql", "sqlite", "mongodb", "redis", "dynamodb", "cassandra", "neo4j", "couchdb",

    # Web Development
    "html", "css", "sass", "react", "next.js", "angular", "vue", "nuxt.js", "svelte",
    "node.js", "express", "nestjs", "fastify", "django", "flask", "fastapi", "laravel",

    # DevOps & Infrastructure
    "git", "github", "gitlab", "bitbucket", "docker", "kubernetes", "terraform", "ansible",
    "vagrant", "jenkins", "circleci", "travisci", "azure devops",

    # Cloud Platforms
    "aws", "azure", "gcp", "heroku", "vercel", "netlify", "digitalocean",

    # Machine Learning & Data
    "tensorflow", "pytorch", "scikit-learn", "xgboost", "lightgbm", "pandas", "numpy", "matplotlib", "seaborn",
    "nltk", "spacy", "opencv", "huggingface", "mlflow", "keras",

    # Big Data & Processing
    "hadoop", "spark", "dask", "kafka", "airflow", "snowflake", "databricks",

    # Scripting & OS
    "bash", "powershell", "linux", "zsh", "windows", "macos",

    # Testing & QA
    "pytest", "unittest", "jest", "mocha", "cypress", "selenium", "jmeter", "postman",

    # Monitoring & Logging
    "elasticsearch", "logstash", "kibana", "prometheus", "grafana", "datadog", "new relic", "sentry",

    # BI & Analytics
    "tableau", "powerbi", "looker", "superset", "metabase", "excel",

    # Tools & Productivity
    "jira", "confluence", "notion", "slack", "figma", "miro", "obsidian"
]

SOFT_SKILLS = [
    "communication", "teamwork", "collaboration", "leadership", "creativity",
    "adaptability", "problem-solving", "time management", "empathy",
    "conflict resolution", "critical thinking", "responsibility", "work ethic",
    "interpersonal skills", "decision making", "negotiation", "patience", "resilience", "self-motivation", "attention to detail",
    "initiative", "reliability", "multitasking", "stress management"
]
