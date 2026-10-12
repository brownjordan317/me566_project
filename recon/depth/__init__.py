"""Component 4, per-frame depth.

grounding fits predicted depth to the sparse points; monocular (MoGe-3) and
multiview (VGGT, patch refinement) predict depth; combine merges object and
background depth; consistency rejects pixels other views see through.
"""
