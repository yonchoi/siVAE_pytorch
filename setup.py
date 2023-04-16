from setuptools import setup,find_packages

long_description = "Protein property prediction model with PLMs for asimov."

setup(
    name="siVAEtorch",  # Required
    version="0.0.1",  # Required
    description="siVAE for pytorch",  # Optional
    packages=find_packages(),  # Required
    package_dir={'siVAEtorch': 'siVAEtorch',},
    python_requires=">=3.8",
    install_requires=['scanpy',
                      'loompy',
                      'torchshape',
                      ],  # Optional
    # include_package_data=True,
    # data_files = [('model', ['./BIVI/models/best_model_MODEL.zip'])],
)
