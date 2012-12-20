#if __STDC__ || __cplusplus
		//In c/c++ the namespace and public should not be included
#else
using System;
namespace SE
{
	[Serializable]
	public
		
#endif
		// IMPORTANT NOTE
		// These ids defines the output data from the Smart Eye system
		// Do not alter any of these ids
	enum SEOutputDataIds
	{
		// Frame information
		SEFrameNumber = 0x0001,
		SEEstimatedDelay = 0x0002,
		SETimeStamp = 0x0003,
		SEUserTimeStamp = 0x0004,
		SEFrameRate = 0x0005,
		SECameraPositions = 0x0006,
		SECameraRotations = 0x0007,
		SEUserDefinedData = 0x0008,
		SERealTimeClock = 0x0009,

		// Head position
		SEHeadPosition = 0x0010,
		SEHeadPositionQ = 0x0011,

		// Head rotation in different formats
		SEHeadRotationRodrigues = 0x0012,
		SEHeadLeftEarDirection = 0x0015,
		SEHeadUpDirection = 0x0014,
		SEHeadNoseDirection = 0x0013,

		SEHeadHeading = 0x0016,
		SEHeadPitch = 0x0017,
		SEHeadRoll = 0x0018,

		SEHeadRotationQ = 0x0019,

		// Raw gaze information
		SEGazeOrigin = 0x001a,
		SELeftGazeOrigin = 0x001b,
		SERightGazeOrigin = 0x001c,

		SEEyePosition = 0x0020,
		SEGazeDirection = 0x0021,
		SEGazeDirectionQ = 0x0022,

		SELeftEyePosition = 0x0023,
		SELeftGazeDirection = 0x0024,
		SELeftGazeDirectionQ = 0x0025,

		SERightEyePosition = 0x0026,
		SERightGazeDirection = 0x0027,
		SERightGazeDirectionQ = 0x0028,

		SEGazeHeading = 0x0029,
		SEGazePitch = 0x002a,

		SELeftGazeHeading = 0x002b,
		SELeftGazePitch = 0x002c,

		SERightGazeHeading = 0x002d,
		SERightGazePitch = 0x002e,


		// Filtered gaze information
		SEFilteredGazeDirection = 0x0030,
		SEFilteredGazeDirectionQ = 0x0031,

		SEFilteredLeftGazeDirection = 0x0032,
		SEFilteredLeftGazeDirectionQ = 0x0033,

		SEFilteredRightGazeDirection = 0x0034,
		SEFilteredRightGazeDirectionQ = 0x0035,

		SEFilteredGazeHeading = 0x0036,
		SEFilteredGazePitch = 0x0037,

		SEFilteredLeftGazeHeading = 0x0038,
		SEFilteredLeftGazePitch = 0x0039,

		SEFilteredRightGazeHeading = 0x003a,
		SEFilteredRightGazePitch = 0x003b,

		SESaccade = 0x003d,
		SEFixation = 0x003e,
		SEBlink = 0x003f,

		// World intersections
		SEClosestWorldIntersection = 0x0040,
		SEFilteredClosestWorldIntersection = 0x0041,

		SEAllWorldIntersections = 0x0042,
		SEFilteredAllWorldIntersections = 0x0043,

		SEZoneId = 0x0044,

		// Eyelid information
		SEEyelidOpening = 0x0050,
		SEEyelidOpeningQ = 0x0051,

		SELeftEyelidOpening = 0x0052,
		SELeftEyelidOpeningQ = 0x0053,

		SERightEyelidOpening = 0x0054,
		SERightEyelidOpeningQ = 0x0055,

		// Keyboard state
		SEKeyboardState = 0x0056,

		// More eyelid information
		SELeftLowerEyelidExtremePoint = 0x0058,
		SELeftUpperEyelidExtremePoint = 0x0059,
		SERightLowerEyelidExtremePoint = 0x005a,
		SERightUpperEyelidExtremePoint = 0x005b,

		// Pupil
		SEPupilDiameter = 0x0060,
		SEPupilDiameterQ = 0x0061,
		SELeftPupilDiameter = 0x0062,
		SELeftPupilDiameterQ = 0x0063,
		SERightPupilDiameter = 0x0064,
		SERightPupilDiameterQ = 0x0065,

		SEFilteredPupilDiameter = 0x0066,
		SEFilteredPupilDiameterQ = 0x0067,
		SEFilteredLeftPupilDiameter = 0x0068,
		SEFilteredLeftPupilDiameterQ = 0x0069,
		SEFilteredRightPupilDiameter = 0x006a,
		SEFilteredRightPupilDiameterQ = 0x006b,

		//GPS 
		SEGPSPosition = 0x0070,
		SEGPSGroundSpeed = 0x0071,
		SEGPSCourse = 0x0072,
		SEGPSTime = 0x0073,

		//Estimated gaze
		SEEstimatedGazeOrigin = 0x007a,
		SEEstimatedLeftGazeOrigin = 0x007b,
		SEEstimatedRightGazeOrigin = 0x007c,

		SEEstimatedEyePosition = 0x0080,
		SEEstimatedGazeDirection = 0x0081,
		SEEstimatedGazeDirectionQ = 0x0082,

		SEEstimatedGazeHeading = 0x0083,
		SEEstimatedGazePitch = 0x0084,

		SEEstimatedLeftEyePosition = 0x0085,
		SEEstimatedLeftGazeDirection = 0x0086,
		SEEstimatedLeftGazeDirectionQ = 0x0087,

		SEEstimatedLeftGazeHeading = 0x0088,
		SEEstimatedLeftGazePitch = 0x0089,

		SEEstimatedRightEyePosition = 0x008a,
		SEEstimatedRightGazeDirection = 0x008b,
		SEEstimatedRightGazeDirectionQ = 0x008c,

		SEEstimatedRightGazeHeading = 0x008d,
		SEEstimatedRightGazePitch = 0x008e,


		SEFilteredEstimatedGazeDirection = 0x0091,
		SEFilteredEstimatedGazeDirectionQ = 0x0092,

		SEFilteredEstimatedGazeHeading = 0x0093,
		SEFilteredEstimatedGazePitch = 0x0094,

		SEFilteredEstimatedLeftGazeDirection = 0x0096,
		SEFilteredEstimatedLeftGazeDirectionQ = 0x0097,

		SEFilteredEstimatedLeftGazeHeading = 0x0098,
		SEFilteredEstimatedLeftGazePitch = 0x0099,

		SEFilteredEstimatedRightGazeDirection = 0x009b,
		SEFilteredEstimatedRightGazeDirectionQ = 0x009c,

		SEFilteredEstimatedRightGazeHeading = 0x009d,
		SEFilteredEstimatedRightGazePitch = 0x009e,

		SELeftDiagnosis = 0x00a0,
		SERightDiagnosis = 0x00a1,
		SELeftGlintsFound = 0x00a2,
		SERightGlintsFound = 0x00a3,

		SEASCIIKeyboardState = 0x00a4,

		SECalibrationGazeIntersection = 0x00b0,
		SETaggedGazeIntersection = 0x00b1,

		SETrackingState = 0x00c0,
		SEEyeglassesStatus = 0x00c1,
		// Driver Id
		SEDriverID = 0x00d0,
		SEDriverIDStatus = 0x00d1,

		SELeftBlinkClosingMidTime = 0x00e0,
		SELeftBlinkOpeningMidTime = 0x00e1,
		SELeftBlinkClosingAmplitude = 0x00e2,
		SELeftBlinkOpeningAmplitude = 0x00e3,
		SELeftBlinkClosingSpeed = 0x00e4,
		SELeftBlinkOpeningSpeed = 0x00e5,

		SERightBlinkClosingMidTime = 0x00e6,
		SERightBlinkOpeningMidTime = 0x00e7,
		SERightBlinkClosingAmplitude = 0x00e8,
		SERightBlinkOpeningAmplitude = 0x00e9,
		SERightBlinkClosingSpeed = 0x00ea,
		SERightBlinkOpeningSpeed = 0x00eb,

		SEChessboardStatus = 0x0200,
		SEChessboardPosition = 0x0201,
		SEChessboardRotation = 0x0202,
		SEAllChessboardStatuses = 0x0205,
		SEAllChessboardPositions = 0x0206,
		SEAllChessboardRotations = 0x0207,

		SEVectorItem = 0x00f0,
		SEStructItem = 0x00f1,
	};
#if __STDC__ || __cplusplus
		//In c/c++ the namespace and public should not be included
#else
}
#endif
