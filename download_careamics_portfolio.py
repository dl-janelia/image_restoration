from pathlib import Path

from careamics_portfolio import PortfolioManager


portfolio = PortfolioManager()
root_path = Path("./data")
portfolio.denoising.N2V_SEM.download(root_path)
portfolio.denoising.CARE_U2OS.download(root_path)
