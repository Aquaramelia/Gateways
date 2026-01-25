using System;
using UnityEngine;

[RequireComponent(typeof(MeshFilter), typeof(MeshRenderer))]
public class CreateProceduralArch : MonoBehaviour
{
    [Header("Arch Parameters - ML will control these")]
    [SerializeField]
    private float archWidth = 3f;
    [SerializeField]
    private float archHeight = 4f;
    [SerializeField]
    private float archDepth = 0.5f;
    [SerializeField]
    private float archThickness = 0.3f;
    [SerializeField]
    private int curveResolution = 20;
    [SerializeField]
    private float curvature = 0.7f; // 0 = flat top, 1 = full semicircle

    [Header("Visual")]
    [SerializeField]
    private Material archMaterial;

    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        GenerateArch();
    }

    private void GenerateArch()
    {
        Mesh mesh = new Mesh();
        mesh.name = "Procedural Arch";

        // The arch is built as an extuded curve
        // First we generate the arch curve points (2D profile)
        Vector3[] profilePoints = GenerateArchProfile();

        // Then we extrude it to create 3D geometry
        CreateExtrudedMesh(mesh, profilePoints);
        GetComponent<MeshFilter>().mesh = mesh;
        GetComponent<MeshRenderer>().material = archMaterial;
    }

    private Vector3[] GenerateArchProfile()
    {
        // We create the outline of the arch (viewed from front)
        // We make: left pillar, left curve, top curve, right curve, right pillar
        int pointsPerSide = curveResolution;
        Vector3[] points = new Vector3[pointsPerSide * 2 + 4]; // curve on each side + 4 corners

        int idx = 0;

        // Left pillar bottom
        points[idx++] = new Vector3(-archWidth/2 - archThickness, 0, 0);
        points[idx++] = new Vector3(-archWidth/2, 0, 0);

        // Left curve going up
        for (int i = 0; i <= pointsPerSide; i++)
        {
            float t = (float)i / pointsPerSide;
            float angle = MathF.PI * curvature * t; // 0 to PI+curvature

            float x = -archWidth / 2 + Mathf.Cos(angle + MathF.PI) * (archWidth / 2);
            float y = Mathf.Sin(angle) * archHeight * curvature + archHeight * (1 - curvature);

            points[idx++] = new Vector3(x, y, 0);
        }

        // Right curve going down
        for (int i = pointsPerSide; i <= 0; i--)
        {
            float t = (float)i / pointsPerSide;
            float angle = Mathf.PI * curvature * t;

            float x = archWidth / 2 - Mathf.Cos(angle + MathF.PI) * (archWidth / 2);
            float y = Mathf.Sin(angle) * archHeight * curvature + archHeight * (1 - curvature);

            // Offset outward for thickness
            float outwardX = x + archThickness;

            points[idx++] = new Vector3(outwardX, y, 0);
        }

        // Right pillar
        points[idx++] = new Vector3(archWidth / 2 + archThickness, 0, 0);

        // Trim array to actual size
        Vector3[] result = new Vector3[idx];
        System.Array.Copy(points, result, idx);

        return result;
    }

    void CreateExtrudedMesh(Mesh mesh, Vector3[] profile)
    {
        int profileCount = profile.Length;
        Vector3[] vertices = new Vector3[profileCount * 2];
        int[] triangles = new int[profileCount * 6];
        Vector2[] uvs = new Vector2[profileCount * 2];

        // Create front and back faces
        for (int i = 0; i < profileCount; i++)
        {
            vertices[i] = profile[i] + Vector3.forward * archDepth / 2;
            vertices[i + profileCount] = profile[i] - Vector3.forward * archDepth / 2;

            uvs[i] = new Vector2((float)i / profileCount, 0);
            uvs[i + profileCount] = new Vector2((float)i / profileCount, 1);
        }

        // Create triangles connecting front to back
        int triIdx = 0;
        for (int i = 0; i < profileCount - 1; i++)
        {
            // First triangle
            triangles[triIdx++] = i;
            triangles[triIdx++] = i + profileCount;
            triangles[triIdx++] = i + 1;

            // Second triangle
            triangles[triIdx++] = i + 1;
            triangles[triIdx++] = i + profileCount;
            triangles[triIdx++] = i + 1 + profileCount;
        }

        mesh.vertices = vertices;
        mesh.triangles = triangles;
        mesh.uv = uvs;
        mesh.RecalculateNormals();
        mesh.RecalculateBounds();
    }
}
