
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <stdio.h>

#include <stdlib.h>
#include "rnnoise.h"
#include "denoise.h"
#include "RtAudio.h"

#include <fcntl.h>
#include <sys/stat.h>

extern "C" {
#include <libavutil/avassert.h>
#include <libavutil/channel_layout.h>
#include <libavutil/opt.h>
#include <libavutil/mathematics.h>
#include <libavutil/timestamp.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
#include <libswresample/swresample.h>
}


#define STREAM_DURATION   10.0
#define STREAM_FRAME_RATE 25 /* 25 images/s */
#define STREAM_PIX_FMT    AV_PIX_FMT_YUV420P /* default pix_fmt */

#define SCALE_FLAGS SWS_BICUBIC

#define FRAME_SIZE 480

typedef short MY_TYPE;
#define READSIZE 1024
#define FORMAT RTAUDIO_SINT16

#define FILENAME_LEN 17
using namespace cv;
using namespace std;

DenoiseState* st;
uint32_t data_size = 0;
float x[FRAME_SIZE];
bool flag = 0;

FILE* fout = NULL, * fout1 = NULL;

int inout(void* /*outputBuffer*/, void* inputBuffer, unsigned int /*nBufferFrames*/,
    double /*streamTime*/, RtAudioStreamStatus status, void* data)
{
    // Since the number of input and output channels is equal, we can do
    // a simple buffer copy operation here.
   // if (status)
     //   std::cout << "Stream over/underflow detected." << std::endl;
    //unsigned int* bytes = (unsigned int*)data;
    short* tmpBufferInput = (short*)inputBuffer;
    if (flag) {
        short tmpBufferOutput[2 * FRAME_SIZE];
        short tmp_input[2 * FRAME_SIZE];
        int i;
        for (i = 0;i < FRAME_SIZE;i++) x[i] = tmpBufferInput[i];
       // rnnoise_process_frame(st, x, x);
        for (i = 0;i < FRAME_SIZE;i++) {
            tmpBufferOutput[2 * i] = x[i];
            tmpBufferOutput[2 * i + 1] = x[i];

            tmp_input[2 * i] = tmpBufferInput[i];
            tmp_input[2 * i + 1] = tmpBufferInput[i];
        }

        fwrite(tmpBufferOutput, sizeof(short), 2 * FRAME_SIZE, fout);
        fwrite(tmp_input, sizeof(short), 2 * FRAME_SIZE, fout1);
        data_size += FRAME_SIZE;
    }
    flag = 1;
    //memcpy(outputBuffer, inputBuffer, 2048);
    return 0;
}

extern "C" {
    // a wrapper around a single output AVStream
    typedef struct OutputStream {
        AVStream* st;
        AVCodecContext* enc;

        /* pts of the next frame that will be generated */
        int64_t next_pts;
        int samples_count;

        AVFrame* frame;
        AVFrame* tmp_frame;

        float t, tincr, tincr2;

        struct SwsContext* sws_ctx;
        struct SwrContext* swr_ctx;
    } OutputStream;

    static void log_packet(const AVFormatContext* fmt_ctx, const AVPacket* pkt)
    {
        AVRational* time_base = &fmt_ctx->streams[pkt->stream_index]->time_base;
        /*
        printf("pts:%s pts_time:%s dts:%s dts_time:%s duration:%s duration_time:%s stream_index:%d\n",
            av_ts2str(pkt->pts), av_ts2timestr(pkt->pts, time_base),
            av_ts2str(pkt->dts), av_ts2timestr(pkt->dts, time_base),
            av_ts2str(pkt->duration), av_ts2timestr(pkt->duration, time_base),
            pkt->stream_index);
            */
    }

    static int write_frame(AVFormatContext* fmt_ctx, AVCodecContext* c,
        AVStream* st, AVFrame* frame)
    {
        int ret;

        // send the frame to the encoder
        ret = avcodec_send_frame(c, frame);
        if (ret < 0) {
            fprintf(stderr, "Error sending a frame to the encoder: \n");
            exit(1);
        }

        while (ret >= 0) {
            AVPacket pkt = { 0 };

            ret = avcodec_receive_packet(c, &pkt);
            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
                break;
            else if (ret < 0) {
                fprintf(stderr, "Error encoding a frame: \n");
                exit(1);
            }

            /* rescale output packet timestamp values from codec to stream timebase */
            av_packet_rescale_ts(&pkt, c->time_base, st->time_base);
            pkt.stream_index = st->index;

            /* Write the compressed frame to the media file. */
            log_packet(fmt_ctx, &pkt);
            ret = av_interleaved_write_frame(fmt_ctx, &pkt);
            av_packet_unref(&pkt);
            if (ret < 0) {
                fprintf(stderr, "Error while writing output packet: \n");
                exit(1);
            }
        }

        return ret == AVERROR_EOF ? 1 : 0;
    }

    /* Add an output stream. */
    static void add_stream(OutputStream* ost, AVFormatContext* oc,
        AVCodec** codec,
        enum AVCodecID codec_id)
    {
        AVCodecContext* c;
        int i;
        AVRational tb_2;
        AVRational tb_3 = { 1, STREAM_FRAME_RATE };

        /* find the encoder */
        *codec = avcodec_find_encoder(codec_id);
        if (!(*codec)) {
            fprintf(stderr, "Could not find encoder for '%s'\n",
                avcodec_get_name(codec_id));
            exit(1);
        }

        ost->st = avformat_new_stream(oc, NULL);
        if (!ost->st) {
            fprintf(stderr, "Could not allocate stream\n");
            exit(1);
        }
        ost->st->id = oc->nb_streams - 1;
        c = avcodec_alloc_context3(*codec);
        if (!c) {
            fprintf(stderr, "Could not alloc an encoding context\n");
            exit(1);
        }
        ost->enc = c;

        switch ((*codec)->type) {
        case AVMEDIA_TYPE_AUDIO:
            c->sample_fmt = (*codec)->sample_fmts ?
                (*codec)->sample_fmts[0] : AV_SAMPLE_FMT_FLTP;
            c->bit_rate = 64000;
            c->sample_rate = 44100;
            if ((*codec)->supported_samplerates) {
                c->sample_rate = (*codec)->supported_samplerates[0];
                for (i = 0; (*codec)->supported_samplerates[i]; i++) {
                    if ((*codec)->supported_samplerates[i] == 44100)
                        c->sample_rate = 44100;
                }
            }
            c->channels = av_get_channel_layout_nb_channels(c->channel_layout);
            c->channel_layout = AV_CH_LAYOUT_STEREO;
            if ((*codec)->channel_layouts) {
                c->channel_layout = (*codec)->channel_layouts[0];
                for (i = 0; (*codec)->channel_layouts[i]; i++) {
                    if ((*codec)->channel_layouts[i] == AV_CH_LAYOUT_STEREO)
                        c->channel_layout = AV_CH_LAYOUT_STEREO;
                }
            }
            c->channels = av_get_channel_layout_nb_channels(c->channel_layout);

            
            tb_2 = { 1, c->sample_rate };
            ost->st->time_base = tb_2;
            break;

        case AVMEDIA_TYPE_VIDEO:
            c->codec_id = codec_id;

            c->bit_rate = 400000;
            /* Resolution must be a multiple of two. */
            c->width = 352;
            c->height = 288;
            /* timebase: This is the fundamental unit of time (in seconds) in terms
             * of which frame timestamps are represented. For fixed-fps content,
             * timebase should be 1/framerate and timestamp increments should be
             * identical to 1. */
            ost->st->time_base = tb_3;
            c->time_base = ost->st->time_base;

            c->gop_size = 12; /* emit one intra frame every twelve frames at most */
            c->pix_fmt = STREAM_PIX_FMT;
            if (c->codec_id == AV_CODEC_ID_MPEG2VIDEO) {
                /* just for testing, we also add B-frames */
                c->max_b_frames = 2;
            }
            if (c->codec_id == AV_CODEC_ID_MPEG1VIDEO) {
                /* Needed to avoid using macroblocks in which some coeffs overflow.
                 * This does not happen with normal video, it just happens here as
                 * the motion of the chroma plane does not match the luma plane. */
                c->mb_decision = 2;
            }
            break;

        default:
            break;
        }

        /* Some formats want stream headers to be separate. */
        if (oc->oformat->flags & AVFMT_GLOBALHEADER)
            c->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
    }

    /**************************************************************/
    /* audio output */

    static AVFrame* alloc_audio_frame(enum AVSampleFormat sample_fmt,
        uint64_t channel_layout,
        int sample_rate, int nb_samples)
    {
        AVFrame* frame = av_frame_alloc();
        int ret;

        if (!frame) {
            fprintf(stderr, "Error allocating an audio frame\n");
            exit(1);
        }

        frame->format = sample_fmt;
        frame->channel_layout = channel_layout;
        frame->sample_rate = sample_rate;
        frame->nb_samples = nb_samples;

        if (nb_samples) {
            ret = av_frame_get_buffer(frame, 0);
            if (ret < 0) {
                fprintf(stderr, "Error allocating an audio buffer\n");
                exit(1);
            }
        }

        return frame;
    }

    static void open_audio(AVFormatContext* oc, AVCodec* codec, OutputStream* ost, AVDictionary* opt_arg)
    {
        AVCodecContext* c;
        int nb_samples;
        int ret;
        AVDictionary* opt = NULL;

        c = ost->enc;

        /* open it */
        av_dict_copy(&opt, opt_arg, 0);
        ret = avcodec_open2(c, codec, &opt);
        av_dict_free(&opt);
        if (ret < 0) {
            fprintf(stderr, "Could not open audio codec: \n");
            exit(1);
        }

        /* init signal generator */
        ost->t = 0;
        ost->tincr = 2 * M_PI * 110.0 / c->sample_rate;
        /* increment frequency by 110 Hz per second */
        ost->tincr2 = 2 * M_PI * 110.0 / c->sample_rate / c->sample_rate;

        if (c->codec->capabilities & AV_CODEC_CAP_VARIABLE_FRAME_SIZE)
            nb_samples = 10000;
        else
            nb_samples = c->frame_size;

        ost->frame = alloc_audio_frame(c->sample_fmt, c->channel_layout,
            c->sample_rate, nb_samples);
        ost->tmp_frame = alloc_audio_frame(AV_SAMPLE_FMT_S16, c->channel_layout,
            c->sample_rate, nb_samples);

        /* copy the stream parameters to the muxer */
        ret = avcodec_parameters_from_context(ost->st->codecpar, c);
        if (ret < 0) {
            fprintf(stderr, "Could not copy the stream parameters\n");
            exit(1);
        }

        /* create resampler context */
        ost->swr_ctx = swr_alloc();
        if (!ost->swr_ctx) {
            fprintf(stderr, "Could not allocate resampler context\n");
            exit(1);
        }

        /* set options */
        av_opt_set_int(ost->swr_ctx, "in_channel_count", c->channels, 0);
        av_opt_set_int(ost->swr_ctx, "in_sample_rate", c->sample_rate, 0);
        av_opt_set_sample_fmt(ost->swr_ctx, "in_sample_fmt", AV_SAMPLE_FMT_S16, 0);
        av_opt_set_int(ost->swr_ctx, "out_channel_count", c->channels, 0);
        av_opt_set_int(ost->swr_ctx, "out_sample_rate", c->sample_rate, 0);
        av_opt_set_sample_fmt(ost->swr_ctx, "out_sample_fmt", c->sample_fmt, 0);

        /* initialize the resampling context */
        if ((ret = swr_init(ost->swr_ctx)) < 0) {
            fprintf(stderr, "Failed to initialize the resampling context\n");
            exit(1);
        }
    }

    /* Prepare a 16 bit dummy audio frame of 'frame_size' samples and
     * 'nb_channels' channels. */
    static AVFrame* get_audio_frame(OutputStream* ost)
    {
        AVFrame* frame = ost->tmp_frame;
        int j, i, v;
        int16_t* q = (int16_t*)frame->data[0];
        AVRational tb_b = { 1, 1 };

        /* check if we want to generate more frames */
        if (av_compare_ts(ost->next_pts, ost->enc->time_base,
            STREAM_DURATION, tb_b) > 0)
            return NULL;

        for (j = 0; j < frame->nb_samples; j++) {
            v = (int)(sin(ost->t) * 10000);
            for (i = 0; i < ost->enc->channels; i++)
                *q++ = v;
            ost->t += ost->tincr;
            ost->tincr += ost->tincr2;
        }

        frame->pts = ost->next_pts;
        ost->next_pts += frame->nb_samples;

        return frame;
    }

    /*
     * encode one audio frame and send it to the muxer
     * return 1 when encoding is finished, 0 otherwise
     */
    static int write_audio_frame(AVFormatContext* oc, OutputStream* ost)
    {
        AVCodecContext* c;
        AVFrame* frame;
        int ret;
        int dst_nb_samples;
        AVRational tb_1;

        c = ost->enc;

        frame = get_audio_frame(ost);

        if (frame) {
            /* convert samples from native format to destination codec format, using the resampler */
            /* compute destination number of samples */
            dst_nb_samples = av_rescale_rnd(swr_get_delay(ost->swr_ctx, c->sample_rate) + frame->nb_samples,
                c->sample_rate, c->sample_rate, AV_ROUND_UP);
            av_assert0(dst_nb_samples == frame->nb_samples);

            /* when we pass a frame to the encoder, it may keep a reference to it
             * internally;
             * make sure we do not overwrite it here
             */
            ret = av_frame_make_writable(ost->frame);
            if (ret < 0)
                exit(1);

            /* convert to destination format */
            ret = swr_convert(ost->swr_ctx,
                ost->frame->data, dst_nb_samples,
                (const uint8_t**)frame->data, frame->nb_samples);
            if (ret < 0) {
                fprintf(stderr, "Error while converting\n");
                exit(1);
            }
            frame = ost->frame;
            tb_1 = { 1, c->sample_rate };
            frame->pts = av_rescale_q(ost->samples_count, tb_1, c->time_base);
            ost->samples_count += dst_nb_samples;
        }

        return write_frame(oc, c, ost->st, frame);
    }

    /**************************************************************/
    /* video output */

    static AVFrame* alloc_picture(enum AVPixelFormat pix_fmt, int width, int height)
    {
        AVFrame* picture;
        int ret;

        picture = av_frame_alloc();
        if (!picture)
            return NULL;

        picture->format = pix_fmt;
        picture->width = width;
        picture->height = height;

        /* allocate the buffers for the frame data */
        ret = av_frame_get_buffer(picture, 0);
        if (ret < 0) {
            fprintf(stderr, "Could not allocate frame data.\n");
            exit(1);
        }

        return picture;
    }

    static void open_video(AVFormatContext* oc, AVCodec* codec, OutputStream* ost, AVDictionary* opt_arg)
    {
        int ret;
        AVCodecContext* c = ost->enc;
        AVDictionary* opt = NULL;

        av_dict_copy(&opt, opt_arg, 0);

        /* open the codec */
        ret = avcodec_open2(c, codec, &opt);
        av_dict_free(&opt);
        if (ret < 0) {
            fprintf(stderr, "Could not open video codec: \n");
            exit(1);
        }

        /* allocate and init a re-usable frame */
        ost->frame = alloc_picture(c->pix_fmt, c->width, c->height);
        if (!ost->frame) {
            fprintf(stderr, "Could not allocate video frame\n");
            exit(1);
        }

        /* If the output format is not YUV420P, then a temporary YUV420P
         * picture is needed too. It is then converted to the required
         * output format. */
        ost->tmp_frame = NULL;
        if (c->pix_fmt != AV_PIX_FMT_YUV420P) {
            ost->tmp_frame = alloc_picture(AV_PIX_FMT_YUV420P, c->width, c->height);
            if (!ost->tmp_frame) {
                fprintf(stderr, "Could not allocate temporary picture\n");
                exit(1);
            }
        }

        /* copy the stream parameters to the muxer */
        ret = avcodec_parameters_from_context(ost->st->codecpar, c);
        if (ret < 0) {
            fprintf(stderr, "Could not copy the stream parameters\n");
            exit(1);
        }
    }

    /* Prepare a dummy image. */
    static void fill_yuv_image(AVFrame* pict, int frame_index,
        int width, int height)
    {
        int x, y, i;

        i = frame_index;

        /* Y */
        for (y = 0; y < height; y++)
            for (x = 0; x < width; x++)
                pict->data[0][y * pict->linesize[0] + x] = x + y + i * 3;

        /* Cb and Cr */
        for (y = 0; y < height / 2; y++) {
            for (x = 0; x < width / 2; x++) {
                pict->data[1][y * pict->linesize[1] + x] = 128 + y + i * 2;
                pict->data[2][y * pict->linesize[2] + x] = 64 + x + i * 5;
            }
        }
    }

    static AVFrame* get_video_frame(OutputStream* ost)
    {
        AVCodecContext* c = ost->enc;
        AVRational tb_1 = { 1,1 };

        /* check if we want to generate more frames */
        if (av_compare_ts(ost->next_pts, c->time_base,
            STREAM_DURATION, tb_1) > 0)
            return NULL;

        /* when we pass a frame to the encoder, it may keep a reference to it
         * internally; make sure we do not overwrite it here */
        if (av_frame_make_writable(ost->frame) < 0)
            exit(1);

        if (c->pix_fmt != AV_PIX_FMT_YUV420P) {
            /* as we only generate a YUV420P picture, we must convert it
             * to the codec pixel format if needed */
            if (!ost->sws_ctx) {
                ost->sws_ctx = sws_getContext(c->width, c->height,
                    AV_PIX_FMT_YUV420P,
                    c->width, c->height,
                    c->pix_fmt,
                    SCALE_FLAGS, NULL, NULL, NULL);
                if (!ost->sws_ctx) {
                    fprintf(stderr,
                        "Could not initialize the conversion context\n");
                    exit(1);
                }
            }
            fill_yuv_image(ost->tmp_frame, ost->next_pts, c->width, c->height);
            sws_scale(ost->sws_ctx, (const uint8_t* const*)ost->tmp_frame->data,
                ost->tmp_frame->linesize, 0, c->height, ost->frame->data,
                ost->frame->linesize);
        }
        else {
            fill_yuv_image(ost->frame, ost->next_pts, c->width, c->height);
        }

        ost->frame->pts = ost->next_pts++;

        return ost->frame;
    }

    /*
     * encode one video frame and send it to the muxer
     * return 1 when encoding is finished, 0 otherwise
     */
    static int write_video_frame(AVFormatContext* oc, OutputStream* ost)
    {
        return write_frame(oc, ost->enc, ost->st, get_video_frame(ost));
    }

    static void close_stream(AVFormatContext* oc, OutputStream* ost)
    {
        avcodec_free_context(&ost->enc);
        av_frame_free(&ost->frame);
        av_frame_free(&ost->tmp_frame);
        sws_freeContext(ost->sws_ctx);
        swr_free(&ost->swr_ctx);
    }

}

uint8_t wav_header[44] = { 'R', 'I', 'F', 'F', 0, 0, 0, 0, 'W', 'A', 'V', 'E', 'f', 'm', 't', 0x20,
    0x10, 0, 0, 0, 0x01, 0, 0x02, 0, 0x80, 0xBB, 0, 0, 0, 0xEE, 0x02, 0, 0x04, 0, 0x10, 0, 'd', 'a', 't', 'a',
    0, 0, 0, 0
};


int main(int argc, char** argv)
{


    OutputStream video_st = { 0 }, audio_st = { 0 };
    const char* filename;
    AVOutputFormat* fmt;
    AVFormatContext* oc;
    AVCodec* audio_codec = NULL, * video_codec = NULL;
    int ret;
    int have_video = 0, have_audio = 0;
    int encode_video = 0, encode_audio = 0;
    AVDictionary* opt = NULL;
    int i;

    if (argc < 2) {
        printf("usage: %s output_file\n"
            "API example program to output a media file with libavformat.\n"
            "This program generates a synthetic audio and video stream, encodes and\n"
            "muxes them into a file named output_file.\n"
            "The output format is automatically guessed according to the file extension.\n"
            "Raw images can also be output by using '%%d' in the filename.\n"
            "\n", argv[0]);
        return 1;
    }

    filename = argv[1];
    for (i = 2; i + 1 < argc; i += 2) {
        if (!strcmp(argv[i], "-flags") || !strcmp(argv[i], "-fflags"))
            av_dict_set(&opt, argv[i] + 1, argv[i + 1], 0);
    }

    /* allocate the output media context */
    avformat_alloc_output_context2(&oc, NULL, NULL, filename);
    if (!oc) {
        printf("Could not deduce output format from file extension: using MPEG.\n");
        avformat_alloc_output_context2(&oc, NULL, "mpeg", filename);
    }
    if (!oc)
        return 1;

    fmt = oc->oformat;

    /* Add the audio and video streams using the default format codecs
     * and initialize the codecs. */
    if (fmt->video_codec != AV_CODEC_ID_NONE) {
        add_stream(&video_st, oc, &video_codec, fmt->video_codec);
        have_video = 1;
        encode_video = 1;
    }
    if (fmt->audio_codec != AV_CODEC_ID_NONE) {
        add_stream(&audio_st, oc, &audio_codec, fmt->audio_codec);
        have_audio = 1;
        encode_audio = 1;
    }

    /* Now that all the parameters are set, we can open the audio and
     * video codecs and allocate the necessary encode buffers. */
    if (have_video)
        open_video(oc, video_codec, &video_st, opt);

    if (have_audio)
        open_audio(oc, audio_codec, &audio_st, opt);

    av_dump_format(oc, 0, filename, 1);

    /* open the output file, if needed */
    if (!(fmt->flags & AVFMT_NOFILE)) {
        ret = avio_open(&oc->pb, filename, AVIO_FLAG_WRITE);
        if (ret < 0) {
            fprintf(stderr, "Could not open '%s'\n", filename);
            return 1;
        }
    }

    /* Write the stream header, if any. */
    ret = avformat_write_header(oc, &opt);
    if (ret < 0) {
        fprintf(stderr, "Error occurred when opening output file: \n");
        return 1;
    }

    while (encode_video || encode_audio) {
        /* select the stream to encode */
        if (encode_video &&
            (!encode_audio || av_compare_ts(video_st.next_pts, video_st.enc->time_base,
                audio_st.next_pts, audio_st.enc->time_base) <= 0)) {
            encode_video = !write_video_frame(oc, &video_st);
        }
        else {
            encode_audio = !write_audio_frame(oc, &audio_st);
        }
    }

    /* Write the trailer, if any. The trailer must be written before you
     * close the CodecContexts open when you wrote the header; otherwise
     * av_write_trailer() may try to use memory that was freed on
     * av_codec_close(). */
    av_write_trailer(oc);

    /* Close each codec. */
    if (have_video)
        close_stream(oc, &video_st);
    if (have_audio)
        close_stream(oc, &audio_st);

    if (!(fmt->flags & AVFMT_NOFILE))
        /* Close the output file. */
        avio_closep(&oc->pb);

    /* free the stream */
    avformat_free_context(oc);

    Mat frame;
    RtAudio adac;

    unsigned int channels, fs, bufferBytes = 0, oDevice = 0, iDevice = 0, iOffset = 0, oOffset = 0;
    unsigned int bufferFrames;
    RtAudio::StreamParameters iParams, oParams;
    RtAudio::StreamOptions options;

    //--- INITIALIZE VIDEOCAPTURE
    VideoCapture cap;
    Mat channel[3];
    
    
    /*struct gwavi_audio_t audio;*/	  /* declare structure used for audio */

    char fourcc[] = "MJPG";		  /* set fourcc used */
    char avi_out[] = "example.avi";    /* set out file name */

    // open the default camera using default API
    // cap.open(0);
    // OR advance usage: select any API backend
    int deviceID = 0;             // 0 = open default camera
    int apiID = cv::CAP_ANY;      // 0 = autodetect default API



    // open selected camera using selected API
    cap.open(deviceID, apiID);
    // check if we succeeded
    if (!cap.isOpened()) {
        cerr << "ERROR! Unable to open camera\n";
        return -1;
    }
    //--- GRAB AND WRITE LOOP
    cout << "Start grabbing" << endl
        << "Press any key to terminate" << endl;

    int frame_width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int frame_height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);

    cap.set(CAP_PROP_FPS, 10);
    int fps = cap.get(CAP_PROP_FPS);

    fout = fopen("C:/Users/NITANSH/Desktop/audio filter project/filtered.wav", "wb");
    fout1 = fopen("C:/Users/NITANSH/Desktop/audio filter project/non_filtered.wav", "wb");



    fwrite(wav_header, sizeof(uint8_t), 44, fout);
    fwrite(wav_header, sizeof(uint8_t), 44, fout1);

    //recordButton->SetLabel("Stop");
    st = rnnoise_create(NULL);


    

    if (adac.getDeviceCount() < 1) {
        // std::cout << "\nNo audio devices found!\n";

        exit(1);
    }

    channels = 1;
    fs = 48000;

    // Let RtAudio print messages to stderr.
    adac.showWarnings(true);

    // Set the same number of channels for both input and output.
    bufferFrames = 480;

    iParams.deviceId = iDevice;
    iParams.nChannels = channels;
    iParams.firstChannel = iOffset;
    oParams.deviceId = oDevice;
    oParams.nChannels = channels;
    oParams.firstChannel = oOffset;

    if (iDevice == 0)
        iParams.deviceId = adac.getDefaultInputDevice();
    if (oDevice == 0)
        oParams.deviceId = adac.getDefaultOutputDevice();


    //options.flags |= RTAUDIO_NONINTERLEAVED;

    try {
        adac.openStream(&oParams, &iParams, FORMAT, fs, &bufferFrames, &inout, (void*)&bufferBytes, &options);
    }
    catch (RtAudioError& e) {
        //std::cout << '\n' << e.getMessage() << '\n' << std::endl;
        exit(1);
    }

    // Test RtAudio functionality for reporting latency.
   // std::cout << "\nStream latency = " << adac.getStreamLatency() << " frames" << std::endl;

    bufferBytes = bufferFrames * channels * sizeof(MY_TYPE);
    
   

    // Define the codec and create VideoWriter object.The output is stored in 'outcpp.avi' file.
    VideoWriter video("outcpp.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 10, Size(frame_width, frame_height));
    int count = 0;
    try {
        adac.startStream(); // start the stream
    }
    catch (RtAudioError& e) {
        //std::cout << '\n' << e.getMessage() << '\n' << std::endl;
        goto cleanup;
    }
    for (;;)
    {
        // wait for a new frame from camera and store it into 'frame'
        cap.read(frame);

        // check if we succeeded
        if (frame.empty()) {
            cerr << "ERROR! blank frame grabbed\n";
            break;
        }

        flip(frame, frame, 1);
        //  int a = frame.elemSize1();
         // int b = frame.depth();
         // int c = frame.channels();

        
        count++;
        video.write(frame);
        // show live and wait for a key with timeout long enough to show images
        imshow("Live", frame);
        if (waitKey(5) >= 0)
            break;
    }

    video.release();
    cap.release();

cleanup:
    if (adac.isStreamOpen()) adac.closeStream();

    uint8_t data_buffer[4] = { 0 };
    data_size = data_size * 4;
    data_size += 36;

    data_buffer[0] = data_size & 0xFF;
    data_buffer[1] = (data_size >> 8) & 0xFF;
    data_buffer[2] = (data_size >> 16) & 0xFF;
    data_buffer[3] = (data_size >> 24) & 0xFF;

    fseek(fout, 4, SEEK_SET);
    fwrite(data_buffer, sizeof(uint8_t), 4, fout);

    fseek(fout1, 4, SEEK_SET);
    fwrite(data_buffer, sizeof(uint8_t), 4, fout1);

    data_size -= 36;

    data_buffer[0] = data_size & 0xFF;
    data_buffer[1] = (data_size >> 8) & 0xFF;
    data_buffer[2] = (data_size >> 16) & 0xFF;
    data_buffer[3] = (data_size >> 24) & 0xFF;

    fseek(fout, 40, SEEK_SET);
    fwrite(data_buffer, sizeof(uint8_t), 4, fout);

    fseek(fout1, 40, SEEK_SET);
    fwrite(data_buffer, sizeof(uint8_t), 4, fout1);

    fclose(fout); // close the output file
    fclose(fout1);

    
    

    // the camera will be deinitialized automatically in VideoCapture destructor
    return 0;

}
