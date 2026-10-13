"""Market data acquisition and preparation.

Downloads and caches daily prices for the fixed 30-stock, 6-sector US universe
(2015-01-01 to 2023-12-31), computes returns, and builds the mean/covariance
inputs consumed by the rest of the pipeline. Paths come from configs/, never code.
"""
