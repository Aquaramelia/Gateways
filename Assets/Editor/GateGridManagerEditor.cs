using UnityEngine;
using UnityEditor;

/// <summary>
/// Custom editor for GateGridManager to add convenient buttons
/// Place this script in an "Editor" folder in your Unity project
/// </summary>
[CustomEditor(typeof(GateGridManager))]
public class GateGridManagerEditor : Editor
{
    public override void OnInspectorGUI()
    {
        DrawDefaultInspector();
        
        GateGridManager manager = (GateGridManager)target;
        
        EditorGUILayout.Space();
        EditorGUILayout.LabelField("Quick Actions", EditorStyles.boldLabel);
        
        if (GUILayout.Button("Generate Maze", GUILayout.Height(30)))
        {
            manager.GenerateMaze();
        }
        
        EditorGUILayout.BeginHorizontal();
        
        if (GUILayout.Button("Random Seed"))
        {
            SerializedProperty seedProp = serializedObject.FindProperty("seed");
            seedProp.intValue = Random.Range(0, 100000);
            serializedObject.ApplyModifiedProperties();
            manager.GenerateMaze();
        }
        
        if (GUILayout.Button("Seed + 1"))
        {
            SerializedProperty seedProp = serializedObject.FindProperty("seed");
            seedProp.intValue++;
            serializedObject.ApplyModifiedProperties();
            manager.GenerateMaze();
        }
        
        EditorGUILayout.EndHorizontal();
        
        EditorGUILayout.Space();
        EditorGUILayout.HelpBox(
            "Tip: Adjust semantic parameters above and click 'Generate Maze' to see different layouts. " +
            "Use different seeds to get unique variations with the same parameters.",
            MessageType.Info
        );
    }
}