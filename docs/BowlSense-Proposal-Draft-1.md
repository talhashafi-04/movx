Proposed Project Title

An Uncertainty-Aware Single-Camera System for Early Screening of Cricket Bowling Actions

It is intended as a smartphone-based cricket coaching and biomechanical screening product for academies, coaches, and developing bowlers.

SDG

9. Industry, Innovation & Infrastructure

The project develops an accessible computer-vision-based sports technology that attempts to make objective biomechanical analysis available using ordinary smartphone hardware rather than specialized motion-capture infrastructure. We have intentionally kept the SDG mapping limited to the one most directly related to the project's technical and innovation contribution.

Category of Project

Image/Video Processing

The core problem is based on computer vision, human pose estimation, temporal video analysis, monocular motion reconstruction, and biomechanical measurement from smartphone video.

We do not intend to classify the project as Generative AI or HealthTech, as neither represents the central technical contribution.

Project Scope

BowlSense is a smartphone-based, markerless cricket-bowling screening system designed for coaches and developing bowlers. A coach records a bowling delivery using a single smartphone positioned in a guided side-on capture configuration, after which the system automatically identifies the delivery interval, detects and temporally tracks relevant body joints, identifies the Arm Horizontal and Ball Release events, and estimates the bowling elbow's extension during the ICC-relevant interval. Because high-speed bowling introduces motion blur, self-occlusion, unusual arm positions, viewpoint variation and monocular depth ambiguity, the system will investigate cricket-specific pose adaptation, temporal motion modelling, biomechanical constraints and uncertainty estimation rather than relying on a simple frame-by-frame pose model. The application will provide a post-delivery skeletal visualization, estimated elbow-extension measurement, confidence/uncertainty information, and a screening recommendation. Low-quality or unreliable recordings will be rejected instead of forcing a potentially misleading result. The system is intended only as an early screening and coaching aid and will not claim to replace formal ICC-accredited biomechanical testing or officially determine whether a bowling action is legal or illegal.

Key Features
Guided Smartphone Capture – Guides the coach regarding camera orientation, framing, body visibility, lighting and supported frame rate before recording.
Automatic Delivery Detection and Clipping – Identifies the bowling action and isolates the relevant delivery segment automatically.
Cricket-Specific Human Pose Estimation – Detects shoulder, elbow, wrist, hip, knee and other relevant joints, with adaptation/fine-tuning for cricket-bowling postures.
Temporal Joint Tracking and Error Correction – Uses consecutive frames to identify unstable keypoints, unrealistic joint jumps, occlusions and motion-related tracking errors.
Monocular Motion/3D Pose Estimation – Investigates temporal 2D-to-3D reconstruction to compensate for the depth information lost by a single RGB camera.
Biomechanical Constraint Enforcement – Uses constraints such as consistent limb lengths, plausible joint motion and temporal continuity to reduce physically implausible pose estimates.
Automatic Arm Horizontal and Ball Release Detection – Detects the two important events required for measuring the relevant elbow-extension interval.
Elbow-Extension Estimation – Calculates the change in bowling-elbow configuration between Arm Horizontal and Ball Release.
Uncertainty-Aware Screening and Failure Rejection – Quantifies measurement reliability and requests a re-recording when the system cannot confidently analyze a delivery.
Coach-Facing Analysis and Session History – Provides post-delivery skeletal overlays, measurements, screening indications and previous delivery/session results for comparison.

The committed scope will focus on bowling only. Batting analysis, medical diagnosis, injury prediction, official legality certification, and generative coaching will remain outside the core scope.

Tools and Technologies

For the technologies section, we propose selecting:

Python
Flutter

Under Other, we propose:

OpenCV
PyTorch
Ultralytics YOLO Pose
MotionBERT / VideoPose3D or an experimentally selected temporal 3D pose model
FastAPI, if a local/server inference API is required
SQLite, for lightweight local session/result storage where required

Complex Computing Problem (CCP)

For the CCP section, we propose the following selections.

1. Range of conflicting requirements
Selected: Involves a variety of factors, which may impose conflicting constraints.

The system must balance biomechanical measurement accuracy against single-camera accessibility, smartphone computational limits, processing latency, ease of camera setup, robustness and the need to reject uncertain measurements.

2. Depth of analysis required
Selected: Has no obvious solution, and requires conceptual thinking and innovative analysis to formulate suitable abstract models.

There is no standard method that guarantees reliable three-dimensional upper-limb biomechanics from one ordinary RGB camera during a high-speed cricket delivery. Alternative pose, temporal, geometric and uncertainty approaches must be experimentally compared.

3. Depth of knowledge required
Selected: A solution requires the use of in-depth computing or domain knowledge and an analytical approach that is based on well-founded principles.

The project requires computer vision, pose estimation, temporal modelling, geometry, monocular 3D reconstruction, uncertainty estimation, video processing and understanding of the relevant cricket biomechanics.

4. Familiarity of issues
Selected: Involves infrequently-encountered issues.

Fast cricket-bowling motion introduces extreme upper-limb configurations, high angular velocities, motion blur, self-occlusion and monocular depth ambiguity that are not typical application-development problems.

5. Level of problem
Selected: Is outside problems encompassed by standards and standard practice for professional computing.

A standard mobile/video-processing pipeline can detect human joints, but determining whether those joints yield sufficiently reliable biomechanical measurements under these conditions is not a standardized software-development problem.

6. Extent of stakeholder involvement and level of conflicting requirements
Selected: Involves several groups of stakeholders with differing and occasionally conflicting needs.

The main stakeholders include bowlers, cricket coaches, sports/biomechanics experts, academies and the project team. Coaches require simple and quick results, while reliable biomechanical analysis requires controlled recording and careful handling of uncertainty.

7. Consequences
Selected: Has consequences that are important locally, but may extend to a broader context.

We believe this is more defensible than selecting the highest consequence level. An incorrect screening result could provide unnecessary concern or false confidence to a bowler or coach, but BowlSense will not make regulatory or medical decisions.

8. Interdependence
Selected: Is a high-level problem possibly including many component parts or sub-problems.

Capture validation, pose estimation, temporal tracking, 3D reconstruction, delivery-event detection, angle estimation, uncertainty modelling and screening are strongly interdependent. Error introduced by an earlier component directly affects the reliability of later outputs.

9. Requirement identification
Selected: Identification of a requirement or the cause of a problem is ill-defined or unknown.

Important requirements such as the minimum usable frame rate, acceptable viewpoint deviation, achievable elbow-angle error, suitable uncertainty threshold and whether full monocular 3D reconstruction is reliable enough cannot be responsibly fixed without experimentation and a proposal-stage POC.

CCP Justification

The Complex Computing Problem in BowlSense is not simply detecting a person or calculating an angle between three pose keypoints. A basic implementation of that nature can be produced using existing pose-estimation libraries, but its output may be unreliable for actual cricket biomechanics. The central problem is determining whether sufficiently accurate upper-limb kinematics can be recovered from a single ordinary smartphone camera during a very fast, partially occluded and three-dimensional bowling movement. The system must therefore reason across noisy temporal pose observations, compensate for monocular depth ambiguity, detect critical delivery events, enforce biomechanically plausible motion, quantify uncertainty and identify when it should refuse to provide a measurement. These components interact directly, and the project requires experimental comparison of alternative methods and validation against independent reference measurements. The system must simultaneously balance measurement reliability, processing latency, ease of use and low-cost deployment, making the underlying problem considerably deeper than routine mobile or computer-vision application development.

We respectfully request approval to replace the current registered topic F26-172-SafeSpec with BowlSense and to update the corresponding title, SDG, category, scope, technologies and CCP information.

Kindly let us know whether we should resubmit the proposal form with these updated details or if any additional approval/documentation is required.

Thank you for your consideration.
Proposed Project Title

An Uncertainty-Aware Single-Camera System for Early Screening of Cricket Bowling Actions

It is intended as a smartphone-based cricket coaching and biomechanical screening product for academies, coaches, and developing bowlers.

SDG

9. Industry, Innovation & Infrastructure

The project develops an accessible computer-vision-based sports technology that attempts to make objective biomechanical analysis available using ordinary smartphone hardware rather than specialized motion-capture infrastructure. We have intentionally kept the SDG mapping limited to the one most directly related to the project's technical and innovation contribution.

Category of Project

Image/Video Processing

The core problem is based on computer vision, human pose estimation, temporal video analysis, monocular motion reconstruction, and biomechanical measurement from smartphone video.

We do not intend to classify the project as Generative AI or HealthTech, as neither represents the central technical contribution.

Project Scope

BowlSense is a smartphone-based, markerless cricket-bowling screening system designed for coaches and developing bowlers. A coach records a bowling delivery using a single smartphone positioned in a guided side-on capture configuration, after which the system automatically identifies the delivery interval, detects and temporally tracks relevant body joints, identifies the Arm Horizontal and Ball Release events, and estimates the bowling elbow's extension during the ICC-relevant interval. Because high-speed bowling introduces motion blur, self-occlusion, unusual arm positions, viewpoint variation and monocular depth ambiguity, the system will investigate cricket-specific pose adaptation, temporal motion modelling, biomechanical constraints and uncertainty estimation rather than relying on a simple frame-by-frame pose model. The application will provide a post-delivery skeletal visualization, estimated elbow-extension measurement, confidence/uncertainty information, and a screening recommendation. Low-quality or unreliable recordings will be rejected instead of forcing a potentially misleading result. The system is intended only as an early screening and coaching aid and will not claim to replace formal ICC-accredited biomechanical testing or officially determine whether a bowling action is legal or illegal.

Key Features
Guided Smartphone Capture – Guides the coach regarding camera orientation, framing, body visibility, lighting and supported frame rate before recording.
Automatic Delivery Detection and Clipping – Identifies the bowling action and isolates the relevant delivery segment automatically.
Cricket-Specific Human Pose Estimation – Detects shoulder, elbow, wrist, hip, knee and other relevant joints, with adaptation/fine-tuning for cricket-bowling postures.
Temporal Joint Tracking and Error Correction – Uses consecutive frames to identify unstable keypoints, unrealistic joint jumps, occlusions and motion-related tracking errors.
Monocular Motion/3D Pose Estimation – Investigates temporal 2D-to-3D reconstruction to compensate for the depth information lost by a single RGB camera.
Biomechanical Constraint Enforcement – Uses constraints such as consistent limb lengths, plausible joint motion and temporal continuity to reduce physically implausible pose estimates.
Automatic Arm Horizontal and Ball Release Detection – Detects the two important events required for measuring the relevant elbow-extension interval.
Elbow-Extension Estimation – Calculates the change in bowling-elbow configuration between Arm Horizontal and Ball Release.
Uncertainty-Aware Screening and Failure Rejection – Quantifies measurement reliability and requests a re-recording when the system cannot confidently analyze a delivery.
Coach-Facing Analysis and Session History – Provides post-delivery skeletal overlays, measurements, screening indications and previous delivery/session results for comparison.

The committed scope will focus on bowling only. Batting analysis, medical diagnosis, injury prediction, official legality certification, and generative coaching will remain outside the core scope.

Tools and Technologies

For the technologies section, we propose selecting:

Python
Flutter

Under Other, we propose:

OpenCV
PyTorch
Ultralytics YOLO Pose
MotionBERT / VideoPose3D or an experimentally selected temporal 3D pose model
FastAPI, if a local/server inference API is required
SQLite, for lightweight local session/result storage where required

Complex Computing Problem (CCP)

For the CCP section, we propose the following selections.

1. Range of conflicting requirements
Selected: Involves a variety of factors, which may impose conflicting constraints.

The system must balance biomechanical measurement accuracy against single-camera accessibility, smartphone computational limits, processing latency, ease of camera setup, robustness and the need to reject uncertain measurements.

2. Depth of analysis required
Selected: Has no obvious solution, and requires conceptual thinking and innovative analysis to formulate suitable abstract models.

There is no standard method that guarantees reliable three-dimensional upper-limb biomechanics from one ordinary RGB camera during a high-speed cricket delivery. Alternative pose, temporal, geometric and uncertainty approaches must be experimentally compared.

3. Depth of knowledge required
Selected: A solution requires the use of in-depth computing or domain knowledge and an analytical approach that is based on well-founded principles.

The project requires computer vision, pose estimation, temporal modelling, geometry, monocular 3D reconstruction, uncertainty estimation, video processing and understanding of the relevant cricket biomechanics.

4. Familiarity of issues
Selected: Involves infrequently-encountered issues.

Fast cricket-bowling motion introduces extreme upper-limb configurations, high angular velocities, motion blur, self-occlusion and monocular depth ambiguity that are not typical application-development problems.

5. Level of problem
Selected: Is outside problems encompassed by standards and standard practice for professional computing.

A standard mobile/video-processing pipeline can detect human joints, but determining whether those joints yield sufficiently reliable biomechanical measurements under these conditions is not a standardized software-development problem.

6. Extent of stakeholder involvement and level of conflicting requirements
Selected: Involves several groups of stakeholders with differing and occasionally conflicting needs.

The main stakeholders include bowlers, cricket coaches, sports/biomechanics experts, academies and the project team. Coaches require simple and quick results, while reliable biomechanical analysis requires controlled recording and careful handling of uncertainty.

7. Consequences
Selected: Has consequences that are important locally, but may extend to a broader context.

We believe this is more defensible than selecting the highest consequence level. An incorrect screening result could provide unnecessary concern or false confidence to a bowler or coach, but BowlSense will not make regulatory or medical decisions.

8. Interdependence
Selected: Is a high-level problem possibly including many component parts or sub-problems.

Capture validation, pose estimation, temporal tracking, 3D reconstruction, delivery-event detection, angle estimation, uncertainty modelling and screening are strongly interdependent. Error introduced by an earlier component directly affects the reliability of later outputs.

9. Requirement identification
Selected: Identification of a requirement or the cause of a problem is ill-defined or unknown.

Important requirements such as the minimum usable frame rate, acceptable viewpoint deviation, achievable elbow-angle error, suitable uncertainty threshold and whether full monocular 3D reconstruction is reliable enough cannot be responsibly fixed without experimentation and a proposal-stage POC.

CCP Justification

The Complex Computing Problem in BowlSense is not simply detecting a person or calculating an angle between three pose keypoints. A basic implementation of that nature can be produced using existing pose-estimation libraries, but its output may be unreliable for actual cricket biomechanics. The central problem is determining whether sufficiently accurate upper-limb kinematics can be recovered from a single ordinary smartphone camera during a very fast, partially occluded and three-dimensional bowling movement. The system must therefore reason across noisy temporal pose observations, compensate for monocular depth ambiguity, detect critical delivery events, enforce biomechanically plausible motion, quantify uncertainty and identify when it should refuse to provide a measurement. These components interact directly, and the project requires experimental comparison of alternative methods and validation against independent reference measurements. The system must simultaneously balance measurement reliability, processing latency, ease of use and low-cost deployment, making the underlying problem considerably deeper than routine mobile or computer-vision application development.

We respectfully request approval to replace the current registered topic F26-172-SafeSpec with BowlSense and to update the corresponding title, SDG, category, scope, technologies and CCP information.

Kindly let us know whether we should resubmit the proposal form with these updated details or if any additional approval/documentation is required.

Thank you for your consideration.
v
