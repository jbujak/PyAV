#include <stdio.h>
#include <npp.h>

#include "libavutil/hwcontext.h"
#include "libavutil/hwcontext_cuda_internal.h"

int convert_and_transfer_nv12_to_bgr24(AVFrame *dst, AVFrame *src) {
    int result = 0;
    CUcontext dummy;
    AVHWFramesContext *ctx = (AVHWFramesContext*)src->hw_frames_ctx->data;
    AVCUDADeviceContext *hwctx = ctx->device_ctx->hwctx;
    CudaFunctions *cu = hwctx->internal->cuda_dl;
    CUdeviceptr tmp;
    CUresult cudaStatus;

    NppiSize size = {
        .width = src->width,
        .height = src->height
    };
    // We pack all three colors into one plane, so it will be 3 times bigger
    int new_linesize = src->linesize[0] * 3;

    cu->cuCtxPushCurrent(hwctx->cuda_ctx);

    // Allocate temporary buffer on GPU
    cudaStatus = cu->cuMemAlloc(&tmp, new_linesize * src->height);
    if (cudaStatus != CUDA_SUCCESS) {
        fprintf(stderr, "cuMemAlloc failed\n");
        result = cudaStatus;
        goto alloc_failed;
    }

    // Perform conversion on GPU
    NppStatus status = nppiNV12ToBGR_8u_P2C3R(
            (Npp8u*)src->data,
            src->linesize[0],
            (Npp8u*)tmp,
            new_linesize,
            size
            );
    if (status != NPP_SUCCESS) {
        fprintf(stderr, "nppiNV12ToBGR_8u_P2C3R failed\n");
        result = status;
        goto error;
    }

    // Wait for conversion to finish
    cudaStatus = cu->cuStreamSynchronize(0);
    if (cudaStatus != CUDA_SUCCESS) {
        fprintf(stderr, "cuStreamSynchronize failed\n");
        result = cudaStatus;
        goto error;
    }

    // Allocate CPU buffer for result
    unsigned char *data_host = malloc(new_linesize * src->height);

    // Copy result from GPU to CPU
    CUDA_MEMCPY2D cpy = {
        .srcMemoryType = CU_MEMORYTYPE_DEVICE,
        .dstMemoryType = CU_MEMORYTYPE_HOST,
        .srcDevice     = (CUdeviceptr)tmp,
        .dstHost       = data_host,
        .srcPitch      = new_linesize,
        .dstPitch      = new_linesize,
        .WidthInBytes  = new_linesize,
        .Height        = src->height,
    };
    cudaStatus = cu->cuMemcpy2DAsync(&cpy, hwctx->stream);
    if (cudaStatus != CUDA_SUCCESS) {
        fprintf(stderr, "cuMemcpy2DAsync failed\n");
        result = cudaStatus;
        goto error;
    }

    // Wait for the transfer to finish
    cudaStatus = cu->cuStreamSynchronize(hwctx->stream);
    if (cudaStatus != CUDA_SUCCESS) {
        fprintf(stderr, "cuStreamSynchronize failed\n");
        result = cudaStatus;
        goto error;
    }

    dst->format = AV_PIX_FMT_BGR24;
    dst->width = src->width;
    dst->height = src->height;
    dst->data[0] = data_host;
    dst->linesize[0] = new_linesize;

error:
    cu->cuMemFree(tmp);

alloc_failed:
    cu->cuCtxPopCurrent(&dummy);
    // Negative value means error
    return (result > 0 ? -result : result);
}
