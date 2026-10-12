"""Fusing several depth models into one agreed depth per pixel.

views chooses neighbouring frames; photometric scores depth hypotheses by how
well full-resolution image patches match across those views, pooled over
regions; filter is the stage entry point that picks, per region, the model
the images support (weighted by each model's measured reliability);
agreement is the earlier rule-based fusion by multi-model, multi-view
agreement, kept as a comparison arm.
"""
