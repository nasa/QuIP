// new test using portaudio

#include <stdio.h>
#include <math.h>
#include "portaudio.h"

typedef struct {
	float left_phase;
	float right_phase;
} paTestData;

PaStream *the_stream;
float arg=0;
float arginc=0;

int sound_callback( const void *inputBuffer, void *outputBuffer,
			unsigned long frameCount, const PaStreamCallbackTimeInfo* timeInfo,
			PaStreamCallbackFlags statusFlags, void *userData )
{
	/* Cast data passed through stream to our structure. */
	paTestData *data = (paTestData*)userData;
	float *out = (float*)outputBuffer;
	unsigned int i;
	(void) inputBuffer; /* Prevent unused variable warning. */
	if( arginc == 0 ) arginc = 8*atan(1)/1000;
	for( i=0; i<frameCount; i++ ) {
#ifdef SAWTOOTH
		*out++ = data->left_phase; /* left */
		*out++ = data->right_phase; /* right */
		/* Generate simple sawtooth phaser that ranges between -1.0 and 1.0. */
		data->left_phase += 0.01f;
		/* When signal reaches top, drop back down. */
		if( data->left_phase >= 1.0f ) data->left_phase -= 2.0f;
		/* higher pitch so we can distinguish left and right. */
		data->right_phase += 0.03f;
		if( data->right_phase >= 1.0f ) data->right_phase -= 2.0f;
#else // ! SAWTOOTH
		*out = *(out+1) = cos(arg);
		arg+= arginc;
#endif // ! SAWTOOTH
	}
	return 0;
}

void pa_end(void)
{
	PaError err;

	err = Pa_Terminate();
	if( err != paNoError ){
		fprintf(stderr, "PortAudio error (Pa_Terminate): %s\n", Pa_GetErrorText( err ) );
		return;
	}
}

#define SAMPLE_RATE (44100)

//static paTestData data;
#define FRAMES_PER_BUFFER	256
static float data[FRAMES_PER_BUFFER*2];

void pa_setup(void)
{
	PaError err;

	err = Pa_Initialize();
	if( err != paNoError ){
		fprintf(stderr, "PortAudio error (Pa_Initialize): %s\n", Pa_GetErrorText( err ) );
		return;
	}

	/* Open an audio I/O stream. */
	err = Pa_OpenDefaultStream( &the_stream,
				0, /* no input channels */
				2, /* stereo output */
				paFloat32, /* 32 bit floating point output */
				SAMPLE_RATE,
				FRAMES_PER_BUFFER, /* frames per buffer, i.e. the number
					of sample frames that PortAudio will
					request from the callback. Many apps
					may want to use
					paFramesPerBufferUnspecified, which
					tells PortAudio to pick the best,
					possibly changing, buffer size.*/
				sound_callback, /* this is your callback function */
				&data ); /*This is a pointer that will be passed to your callback*/
	if( err != paNoError ){
		fprintf(stderr, "PortAudio error (Pa_OpenDefaultStream): %s\n", Pa_GetErrorText( err ) );
		pa_end();
		return;
	}

	err = Pa_StartStream( the_stream );
	if( err != paNoError ) {
		fprintf(stderr, "PortAudio error (Pa_StartStream): %s\n", Pa_GetErrorText( err ) );
		pa_end();
		return;
	}
}

void pa_stop(void)
{
	PaError err;

	err = Pa_StopStream( the_stream );
	if( err != paNoError ) {
		fprintf(stderr, "PortAudio error (Pa_StopStream): %s\n", Pa_GetErrorText( err ) );
		pa_end();
		return;
	}

	err = Pa_CloseStream( stream );
	if( err != paNoError ) {
		fprintf(stderr, "PortAudio error (Pa_CloseStream): %s\n", Pa_GetErrorText( err ) );
		pa_end();
		return;
	}

}

int main(int ac, char **av)
{
	pa_setup();

	Pa_Sleep(4000);	// milliseconds

	pa_stop();
	pa_end();
}

