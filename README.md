$header = @"
---
title: PII Compliance Auditor
emoji: 🔒
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
tags:
  - openenv
  - rl
  - pii
  - compliance
  - agent
---

"@

$body = Get-Content README.md -Raw
$header + $body | Set-Content README.md -Encoding UTF8

Get-Content README.md -TotalCount 12