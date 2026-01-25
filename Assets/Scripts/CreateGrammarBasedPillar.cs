using UnityEngine;
using System.Collections.Generic;
using System.Text;
using System;

public class CreateGrammarBasedPillar : MonoBehaviour
{
    [Header("L-System Rules - ML controls these")]
    [SerializeField]
    private string axiom = "F";
    [SerializeField]
    private string ruleF = "F[+F]F[-F]F"; // ML could generate variations
    [SerializeField]
    private int iterations = 3;

    [Header("Interpretation Parameters - ML controls these")]
    [SerializeField]
    private float segmentLength = 0.5f;
    [SerializeField]
    private float angle = 25f;
    [SerializeField]
    private float thickness = 0.1f;
    [SerializeField]
    private float thicknessDecay = 0.8f;

    [Header("Visual")]
    [SerializeField]
    private Material lineMaterial;

    void Start()
    {
        GeneratePillar();
    }

    private void GeneratePillar()
    {
        string lSystem = GenerateLSystem();
        InterpretLSystem(lSystem);
    }

    private void InterpretLSystem(string lSystem)
    {
        // Turtle graphics interpretation
        Vector3 position = transform.position;
        Quaternion rotation = Quaternion.identity;
        float currentThickness = thickness;

        Stack<TurtleState> stateStack = new Stack<TurtleState>();

        foreach(char c in lSystem)
        {
            switch (c)
            {
                case 'F': // Move forward and draw
                    Vector3 newPosition = position + rotation * Vector3.up * segmentLength;
                    CreateSegment(position, newPosition, currentThickness);
                    position = newPosition;
                    break;

                case '+': // Rotate right
                    rotation *= Quaternion.Euler(0, 0, -angle);
                    break;
                    
                case '-': // Rotate left
                    rotation *= Quaternion.Euler(0, 0, angle);
                    break;

                case '[': // Push state
                    stateStack.Push(new TurtleState(position, rotation, currentThickness));
                    currentThickness *= thicknessDecay;
                    break;
                    
                case ']': // Pop state
                    if (stateStack.Count > 0)
                    {
                        TurtleState state = stateStack.Pop();
                        position = state.position;
                        rotation = state.rotation;
                        currentThickness = state.thickness;
                    }
                    break;
            }
        }
    }

    private void CreateSegment(Vector3 start, Vector3 end, float thickness)
    {
        GameObject segment = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
        segment.transform.parent = transform;

        // Position and orient cylinder
        Vector3 midpoint = (start + end) / 2f;
        segment.transform.position = midpoint;

        Vector3 direction = end - start;
        segment.transform.up = direction;

        float length = direction.magnitude;
        segment.transform.localScale = new Vector3(thickness, length / 2f, thickness);

        if (lineMaterial != null)
        {
            segment.GetComponent<Renderer>().material = lineMaterial;
        }

        // Remove collider if you don't need physics 
        // Destroy(segment.GetComponent<Collider>());
    }

    private string GenerateLSystem()
    {
        string current = axiom;
        for (int i = 0; i < iterations; i ++)
        {
            StringBuilder next = new StringBuilder();

            foreach (char c in current)
            {
                switch(c)
                {
                    case 'F':
                        next.Append(ruleF);
                        break;
                    default:
                        next.Append(c);
                        break;
                }
            }

            current = next.ToString();
        }

        Debug.Log($"L-System generated: {current.Substring(0, Mathf.Min(100, current.Length))}...");
        return current;
    }
}

struct TurtleState
{
    public Vector3 position;
    public Quaternion rotation;
    public float thickness;
    
    public TurtleState(Vector3 pos, Quaternion rot, float thick)
    {
        position = pos;
        rotation = rot;
        thickness = thick;
    }
}