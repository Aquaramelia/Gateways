# Gateways

**Status: Early Development** 🚧

A procedural generation system that transforms words into experiential maze environments. Using machine learning to map semantic meaning to architectural parameters, Gateways creates unique pillar and gateway structures in Unity that evoke the emotional and conceptual qualities of input text.

## Overview

Gateways bridges natural language processing and procedural generation to create immersive spatial experiences. Enter a word, and watch as a machine learning model translates its semantic properties into architectural parameters that define the structure, decoration, and atmosphere of a traversable maze.

## Concept

1. **Input**: User provides a word (e.g., "serenity", "chaos", "ancient")
2. **ML Processing**: Trained model generates architectural parameters based on semantic analysis
3. **Parameter Export**: Generated values saved to JSON/CSV file
4. **Unity Import**: Unity reads parameter file
5. **Procedural Generation**: Maze elements (pillars, gateways, decorations) generated according to parameters
6. **Experience**: User navigates the maze in play mode

## Technical Stack

- **Machine Learning**: [TBD - TensorFlow/PyTorch/scikit-learn]
- **Game Engine**: Unity
- **Data Format**: JSON/CSV parameter files
- **Languages**: Python (ML), C# (Unity)

## Project Structure

```
Gateways/
├── Assets/                     # Unity project files
├── Assets/PythonScripts        # Machine learning training and inference
```

## Why This Project?

This project explores the intersection of language, perception, and space - investigating how semantic meaning can be translated into architectural form to create emotionally resonant environments. It's an experiment in computational creativity and experiential design.

---

*Gateways: Where language shapes space.*