# setup-blog.ps1
# Run from the root of your Hugo repo: assoudi-blog

$ErrorActionPreference = "Stop"

Write-Host "Starting Hugo blog setup..." -ForegroundColor Cyan

# --- Sanity checks ---
if (-not (Test-Path ".git")) {
    throw "This folder is not a Git repository. Run this from the root of your assoudi-blog repo."
}

if (-not (Get-Command hugo -ErrorAction SilentlyContinue)) {
    throw "Hugo is not installed or not in PATH."
}

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw "Git is not installed or not in PATH."
}

# --- Ensure content directories exist ---
$dirs = @(
    "content",
    "content\posts",
    "static",
    "layouts",
    "assets"
)

foreach ($dir in $dirs) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir | Out-Null
        Write-Host "Created directory: $dir"
    }
}

# --- Add PaperMod theme if missing ---
if (-not (Test-Path "themes\PaperMod")) {
    Write-Host "Adding PaperMod theme as git submodule..." -ForegroundColor Yellow
    git submodule add https://github.com/adityatelange/hugo-PaperMod.git themes/PaperMod
} else {
    Write-Host "PaperMod theme already exists."
}

git submodule update --init --recursive

# --- Write hugo.toml ---
$hugoToml = @"
baseURL = 'https://assoudi.blog/'
languageCode = 'en-us'
title = 'Hicham Assoudi'
theme = 'PaperMod'

[pagination]
  pagerSize = 10

[params]
  env = "production"
  title = "Hicham Assoudi"
  description = "Oracle AI, vector search, ONNX workflows, architecture patterns, and benchmark-driven enterprise AI writing."
  author = "Hicham Assoudi"
  ShowReadingTime = true
  ShowShareButtons = true
  ShowPostNavLinks = true
  ShowBreadCrumbs = true
  ShowCodeCopyButtons = true
  defaultTheme = "auto"

[params.homeInfoParams]
  Title = "Oracle AI in Practice"
  Content = "Practical writing on Oracle AI, vector search, ONNX workflows, enterprise architecture, and benchmark-driven implementation patterns."

[[menu.main]]
  identifier = "articles"
  name = "Articles"
  url = "/posts/"
  weight = 10

[[menu.main]]
  identifier = "about"
  name = "About"
  url = "/about/"
  weight = 20

[[menu.main]]
  identifier = "books"
  name = "Books"
  url = "/books/"
  weight = 30

[[menu.main]]
  identifier = "resources"
  name = "Resources"
  url = "/resources/"
  weight = 40
"@

Set-Content -Path "hugo.toml" -Value $hugoToml -Encoding UTF8
Write-Host "Created/updated hugo.toml"

# --- Write content files ---
$welcomePost = @"
---
title: "Welcome to assoudi.blog"
date: 2026-04-05
draft: false
description: "Why I launched assoudi.blog and what I will publish here."
tags: ["oracle", "ai", "vector-search"]
categories: ["articles"]
---

Welcome to assoudi.blog.

This is where I publish practical, technically grounded writing on Oracle AI, vector search, ONNX workflows, enterprise architecture, and benchmark-driven implementation patterns.

My goal is simple: useful Oracle + AI content for practitioners, architects, and technical leaders.
"@

$aboutPage = @"
---
title: "About"
date: 2026-04-05
draft: false
---

I’m Hicham Assoudi, an Oracle architect, AI researcher, founder of Typica.ai, and technical author.

I write about practical Oracle AI, vector search, ONNX workflows, architecture patterns, and enterprise implementation trade-offs.
"@

$booksPage = @"
---
title: "Books"
date: 2026-04-05
draft: false
---

This page will list my books, technical writing, and related publication resources.
"@

$resourcesPage = @"
---
title: "Resources"
date: 2026-04-05
draft: false
---

This page will gather useful links to GitHub repositories, Hugging Face assets, videos, and companion resources.
"@

Set-Content -Path "content\posts\welcome-to-assoudi-blog.md" -Value $welcomePost -Encoding UTF8
Set-Content -Path "content\about.md" -Value $aboutPage -Encoding UTF8
Set-Content -Path "content\books.md" -Value $booksPage -Encoding UTF8
Set-Content -Path "content\resources.md" -Value $resourcesPage -Encoding UTF8

Write-Host "Created content pages"

# --- Git status and first commit ---
git add .

# Commit only if there are staged changes
$gitDiff = git diff --cached --name-only
if ($gitDiff) {
    git commit -m "Initial Hugo blog setup"
    Write-Host "Committed initial setup" -ForegroundColor Green
} else {
    Write-Host "No changes to commit."
}

Write-Host ""
Write-Host "Setup complete." -ForegroundColor Green
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. Run: hugo server"
Write-Host "2. Open: http://localhost:1313/"
Write-Host "3. If it looks good, run: git push -u origin main"