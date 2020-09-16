---
title: Smart Aquaculture
summary: Smart Aquaculture Monitoring System
tags:
- All
date: "2020-09-16T00:00:00Z"

# Optional external URL for project (replaces project detail page).
external_link: ""

image:
  caption: Photo by rawpixel on Unsplash
  focal_point: Smart

links:
- icon: linkedin
  icon_pack: fab
  name: Connect on 
  url: https://www.linkedin.com/in/ssuppuluri/
url_code: ""
url_pdf: ""
url_slides: ""
url_video: ""

# Slides (optional).
#   Associate this project with Markdown slides.
#   Simply enter your slide deck's filename without extension.
#   E.g. `slides = "example-slides"` references `content/slides/example-slides.md`.
#   Otherwise, set `slides = ""`.
slides: example
---

Smart Aquaculture presents a Monitoring system, to monitor the analog parameters and transmit these values to the base station where they can be read and compared with the set points. If these values exceed their corresponding set points a short message will be sent to the owners mobile through Global System for Mobile (GSM). This system also uses ZigBee to implement this application.

The analog parameters like pH, Temperature and Humidity are read by the respective sensors and these values are transmitted to the base station through ZigBee wireless communication. The base station receives these values and passes the data to the controller section. The ARM controller compares these values with the fixed values and if they exceed the set points, the ARM controller displays the parameter, which actually exceeds its set point, on the LCD with short message alert to owners.