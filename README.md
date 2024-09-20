# Unified-SD-Pipeline

A unified Stable Diffusion pipeline based on [*diffusers*](https://github.com/huggingface/diffusers)

## Overview

This repo is an integration of multiple Stable Diffusion pipelines in *diffusers*

The pipelines are modified and called in a deconstructed manner, which is similar to the official [tutorial of pipeline deconstruction](https://huggingface.co/docs/diffusers/using-diffusers/write_own_pipeline) by *diffusers*

## Supported Pipelines

### [diffusers.StableDiffusionPipeline](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/text2img#diffusers.StableDiffusionPipeline)

- [CompVis/stable-diffusion-v1-4](https://huggingface.co/CompVis/stable-diffusion-v1-4)
- runwayml/stable-diffusion-v1-5 (deprecated)
- [stabilityai/stable-diffusion-2-1](https://huggingface.co/stabilityai/stable-diffusion-2-1)

### [diffusers.StableDiffusionXLPipeline](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl#diffusers.StableDiffusionXLPipeline)

- [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) with [stabilityai/stable-diffusion-xl-refiner-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0)

### [diffusers.StableDiffusion3Pipeline](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_3#diffusers.StableDiffusion3Pipeline)

- [stabilityai/stable-diffusion-3-medium-diffusers](https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers)

### [diffusers.FluxPipeline](https://huggingface.co/docs/diffusers/api/pipelines/flux#diffusers.FluxPipeline)

- [black-forest-labs/FLUX.1-schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell)

## Usage

Run `./generate.sh` to generate a set of images according to given settings

Modify `generate.sh` for customized experimental settings

