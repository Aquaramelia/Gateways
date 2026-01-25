using UnityEngine;
using System.Collections.Generic;
using System.Text;

public class CreateGrammarBasedGateway : MonoBehaviour
{
    [Header("L-System Rules - ML controls these")]
    [Tooltip("Starting symbol")]
    public string axiom = "A";
    
    [Tooltip("Rule for A: creates the main arch structure")]
    public string ruleA = "F[+A]F[-A]FA";
    
    [Tooltip("Rule for F: can add ornamentation")]
    public string ruleF = "FF";
    
    public int iterations = 3;
    
    [Header("Gateway Shape Parameters - ML controls these")]
    [Tooltip("Base segment length")]
    public float segmentLength = 0.5f;
    
    [Tooltip("Branching angle in degrees")]
    public float angle = 25f;
    
    [Tooltip("How much the arch curves inward at the top")]
    public float archCurveAngle = 15f;
    
    [Tooltip("Starting thickness of segments")]
    public float thickness = 0.15f;
    
    [Tooltip("How much thinner branches become")]
    public float thicknessDecay = 0.7f;
    
    [Tooltip("Makes structure grow upward more")]
    public float upwardBias = 0.3f;
    
    [Header("Visual")]
    public Material segmentMaterial;
    public Color baseColor = Color.white;
    
    [Header("Symmetry")]
    [Tooltip("Create mirrored copy for symmetrical gateway")]
    public bool createMirror = true;

    void Start()
    {
        GenerateGateway();
    }

    [ContextMenu("Regenerate Gateway")]
    public void GenerateGateway()
    {
        // Clear existing children
        foreach (Transform child in transform)
        {
            Destroy(child.gameObject);
        }
        
        string lSystem = GenerateLSystem();
        
        // Generate left pillar/side
        GameObject leftSide = new GameObject("LeftSide");
        leftSide.transform.parent = transform;
        leftSide.transform.localPosition = Vector3.left * 1.5f; // Offset to left
        InterpretLSystem(lSystem, leftSide.transform);
        
        // Generate right pillar/side (mirrored)
        if (createMirror)
        {
            GameObject rightSide = new GameObject("RightSide");
            rightSide.transform.parent = transform;
            rightSide.transform.localPosition = Vector3.right * 1.5f; // Offset to right
            rightSide.transform.localScale = new Vector3(-1, 1, 1); // Mirror on X axis
            InterpretLSystem(lSystem, rightSide.transform);
        }
    }

    string GenerateLSystem()
    {
        string current = axiom;
        
        for (int i = 0; i < iterations; i++)
        {
            StringBuilder next = new StringBuilder();
            
            foreach (char c in current)
            {
                switch (c)
                {
                    case 'A':
                        next.Append(ruleA);
                        break;
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
        
        Debug.Log($"L-System: {current.Substring(0, Mathf.Min(200, current.Length))}...");
        return current;
    }

    void InterpretLSystem(string lSystem, Transform parent)
    {
        Vector3 position = Vector3.zero;
        Vector3 direction = Vector3.up; // Start going upward
        float currentThickness = thickness;
        int depth = 0;
        
        Stack<TurtleState> stateStack = new Stack<TurtleState>();
        
        foreach (char c in lSystem)
        {
            switch (c)
            {
                case 'F': // Move forward and draw
                    Vector3 newPosition = position + direction * segmentLength;
                    CreateSegment(position, newPosition, currentThickness, depth, parent);
                    position = newPosition;
                    break;
                    
                case 'A': // Special case: just marks branching points, no draw
                    break;
                    
                case '+': // Rotate counter-clockwise
                    direction = Quaternion.Euler(0, 0, angle + archCurveAngle * (position.y / 5f)) * direction;
                    break;
                    
                case '-': // Rotate clockwise  
                    direction = Quaternion.Euler(0, 0, -angle - archCurveAngle * (position.y / 5f)) * direction;
                    break;
                    
                case '^': // Rotate upward (3D)
                    direction = Quaternion.Euler(upwardBias * 10, 0, 0) * direction;
                    break;
                    
                case '&': // Rotate downward (3D)
                    direction = Quaternion.Euler(-upwardBias * 10, 0, 0) * direction;
                    break;
                    
                case '[': // Push state
                    stateStack.Push(new TurtleState(position, direction, currentThickness, depth));
                    currentThickness *= thicknessDecay;
                    depth++;
                    break;
                    
                case ']': // Pop state
                    if (stateStack.Count > 0)
                    {
                        TurtleState state = stateStack.Pop();
                        position = state.position;
                        direction = state.direction;
                        currentThickness = state.thickness;
                        depth = state.depth;
                    }
                    break;
            }
        }
    }

    void CreateSegment(Vector3 start, Vector3 end, float segmentThickness, int depth, Transform parent)
    {
        GameObject segment = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
        segment.transform.parent = parent;
        segment.name = $"Segment_D{depth}";
        
        // Position and orient
        Vector3 midpoint = (start + end) / 2f;
        segment.transform.position = parent.position + midpoint;
        
        Vector3 segmentDirection = end - start;
        if (segmentDirection.magnitude > 0.001f)
        {
            segment.transform.rotation = Quaternion.FromToRotation(Vector3.up, segmentDirection);
        }
        
        float length = segmentDirection.magnitude;
        segment.transform.localScale = new Vector3(segmentThickness, length / 2f, segmentThickness);
        
        // Material and color
        Renderer renderer = segment.GetComponent<Renderer>();
        if (segmentMaterial != null)
        {
            renderer.material = segmentMaterial;
        }
        
        // Color gradient by depth (optional: makes structure more readable)
        float depthFactor = depth / 5f;
        renderer.material.color = Color.Lerp(baseColor, baseColor * 0.6f, depthFactor);
        
        // Remove collider
        Destroy(segment.GetComponent<Collider>());
    }

    struct TurtleState
    {
        public Vector3 position;
        public Vector3 direction;
        public float thickness;
        public int depth;
        
        public TurtleState(Vector3 pos, Vector3 dir, float thick, int d)
        {
            position = pos;
            direction = dir;
            thickness = thick;
            depth = d;
        }
    }
}