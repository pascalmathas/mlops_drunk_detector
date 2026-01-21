#!/bin/bash

curl -X POST http://localhost:5001/predict \
-H "Content-Type: application/json" \
-d '{
           "x": [5.1, 6.2, 7.3, 8.4, 9.5],
           "y": [0.5, 0.4, 0.3, 0.2, 0.1],
           "z": [0.9, 0.8, 0.9, 0.8, 0.9],
           "time": [0, 10, 20, 30, 40],
           "pid": "BK7610",
           "phonetype": "iPhone"
         }'

curl -X POST http://localhost:5001/predict \
-H "Content-Type: application/json" \
-d '{
           "x": [10.1, 0.2, 0.3, 0.4, 0.5],
           "y": [0.5, 0.4, 0.3, 0.2, 0.1],
           "z": [0.9, 0.8, 0.9, 0.8, 0.9],
           "time": [0, 10, 20, 30, 40],
           "pid": "BK7611",
           "phonetype": "Samsung Galaxy"
         }'

curl -X POST http://localhost:5001/predict \
-H "Content-Type: application/json" \
-d '{
           "x": [-5.5, 0.2, 0.3, 0.4, 0.5],
           "y": [0.5, 0.4, 0.3, 0.2, 0.1],
           "z": [0.9, 0.8, 0.9, 0.8, 0.9],
           "time": [0, 10, 20, 30, 40],
           "pid": "BK7612",
           "phonetype": "Google Pixel"
         }'

curl -X POST http://localhost:5001/predict \
-H "Content-Type: application/json" \
-d '{
           "x": [8.8, 0.2, 0.3, 0.4, 0.5],
           "y": [0.5, 0.4, 0.3, 0.2, 0.1],
           "z": [0.9, 0.8, 0.9, 0.8, 0.9],
           "time": [0, 10, 20, 30, 40],
           "pid": "BK7613",
           "phonetype": "Xiaomi"
         }'

curl -X POST http://localhost:5001/predict \
-H "Content-Type: application/json" \
-d '{
           "x": [12.3, 0.2, 0.3, 0.4, 0.5],
           "y": [0.5, 0.4, 0.3, 0.2, 0.1],
           "z": [0.9, 0.8, 0.9, 0.8, 0.9],
           "time": [0, 10, 20, 30, 40],
           "pid": "BK7614",
           "phonetype": "OnePlus"
         }'

curl -X POST http://localhost:5001/predict \
-H "Content-Type: application/json" \
-d '{
           "x": [0.1, 0.2, 0.3, 0.4, 0.5],
           "y": [20.5, 25.4, 30.3, 0.2, 0.1],
           "z": [0.9, 0.8, 0.9, 0.8, 0.9],
           "time": [0, 10, 20, 30, 40],
           "pid": "BK7615",
           "phonetype": "iPhone"
         }'

curl -X POST http://localhost:5001/predict \
-H "Content-Type: application/json" \
-d '{
           "x": [0.1, 0.2, 0.3, 0.4, 0.5],
           "y": [-15.5, 0.4, 0.3, 0.2, 0.1],
           "z": [0.9, 0.8, 0.9, 0.8, 0.9],
           "time": [0, 10, 20, 30, 40],
           "pid": "BK7616",
           "phonetype": "Samsung Galaxy"
         }'

curl -X POST http://localhost:5001/predict \
-H "Content-Type: application/json" \
-d '{
           "x": [0.1, 0.2, 0.3, 0.4, 0.5],
           "y": [0.5, 0.4, 50.3, 0.2, 0.1],
           "z": [0.9, 0.8, 0.9, 0.8, 0.9],
           "time": [0, 10, 20, 30, 40],
           "pid": "BK7617",
           "phonetype": "Google Pixel"
         }'

curl -X POST http://localhost:5001/predict \
-H "Content-Type: application/json" \
-d '{
           "x": [0.1, 0.2, 0.3, 0.4, 0.5],
           "y": [0.5, 0.4, 0.3, 0.2, 0.1],
           "z": [15.9, 18.8, 0.9, 0.8, 0.9],
           "time": [0, 10, 20, 30, 40],
           "pid": "BK7618",
           "phonetype": "Xiaomi"
         }'

curl -X POST http://localhost:5001/predict \
-H "Content-Type: application/json" \
-d '{
           "x": [0.1, 0.2, 0.3, 0.4, 0.5],
           "y": [0.5, 0.4, 0.3, 0.2, 0.1],
           "z": [0.9, -20.8, 0.9, 0.8, 0.9],
           "time": [0, 10, 20, 30, 40],
           "pid": "BK7619",
           "phonetype": "OnePlus"
         }'

curl -X POST http://localhost:5001/predict \
-H "Content-Type: application/json" \
-d '{
           "x": [0.1, 0.2, 0.3, 0.4, 0.5],
           "y": [0.5, 0.4, 0.3, 0.2, 0.1],
           "z": [0.9, 0.8, 0.9, 0.8, 0.9],
           "pid": "BK7619",
           "phonetype": "iPhone"
         }'

curl -X POST http://localhost:5001/predict \
-H "Content-Type: application/json" \
-d '{
           "x": [0.2, 0.3, 0.4],
           "y": [0.6, 0.5, 0.4],
           "z": [0.85, 0.86, 0.87],
           "pid": "BK7619",
           "phonetype": "Samsung Galaxy"
         }'

curl -X POST http://localhost:5001/predict \
-H "Content-Type: application/json" \
-d '{
           "x": [0.1, 0.2, 0.3, 0.4, 0.5],
           "y": [0.5, 0.4, 0.3, 0.2, 0.1],
           "z": [0.9, 0.8, 0.9, 0.8, 0.9],
           "time": [0, 10, 20],
           "pid": "BK7619",
           "phonetype": "Google Pixel"
         }'

curl -X POST http://localhost:5001/predict \
-H "Content-Type: application/json" \
-d '{
           "x": [0.15, 0.25],
           "y": [0.55, 0.45],
           "z": [0.91, 0.92],
           "time": [0, 10, 20, 30, 40],
           "pid": "BK7619",
           "phonetype": "OnePlus"
         }'

curl -X POST http://localhost:5001/predict \
-H "Content-Type: application/json" \
-d '{
           "x": [0.1],
           "y": [0.5],
           "z": [0.9],
           "time": [0, 10, 20, 30, 40, 50],
           "pid": "BK7619",
           "phonetype": "Xiaomi"
         }'
