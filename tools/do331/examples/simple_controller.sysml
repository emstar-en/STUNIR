/* STUNIR DO-331 Model-Based Development Output
 * Schema Version: 1
 * Source IR Hash: sha256:example
 * Generation Epoch: 1738108800
 * DAL Level: B
 */

import ScalarValues::*;

package AltitudeMonitor {
    doc /* Module: AltitudeMonitor */

    attribute def AltitudeT :> Real {
        doc /* Type: altitude_t (feet) */
    }

    attribute def SpeedT :> Real {
        doc /* Type: speed_t (knots) */
    }

    action def CheckAltitude {
        in currentAlt : AltitudeT;
        in threshold : AltitudeT;
        out result : Boolean;

        // DO-331 Coverage Point: CP_ENTRY_checkAltitude (entry)
        first start;

        then if currentAlt > threshold [
            // DO-331 Coverage Point: CP_DEC_1_T (decision_true)
            then assign result := true;
        ] else [
            // DO-331 Coverage Point: CP_DEC_1_F (decision_false)
            then assign result := false;
        ]

        // DO-331 Coverage Point: CP_EXIT_checkAltitude (exit)
        then done;
    }

    action def GetStatus {
        in altitude : AltitudeT;
        in speed : SpeedT;
        out status : String;  // "ok", "warning", "critical"

        // DO-331 Coverage Point: CP_ENTRY_getStatus (entry)
        first start;

        then if altitude > 40000.0 [
            // DO-331 Coverage Point: CP_DEC_2_T (decision_true)
            then assign status := "critical";
        ] else [
            // DO-331 Coverage Point: CP_DEC_2_F (decision_false)
            then if altitude > 35000.0 [
                // DO-331 Coverage Point: CP_DEC_3_T (decision_true)
                then assign status := "warning";
            ] else [
                // DO-331 Coverage Point: CP_DEC_3_F (decision_false)
                then assign status := "ok";
            ]
        ]

        // DO-331 Coverage Point: CP_EXIT_getStatus (exit)
        then done;
    }

    state def FlightMode {
        // DO-331 Coverage Point: CP_STATE_ground (state_coverage)
        entry state Ground;
        
        // DO-331 Coverage Point: CP_STATE_takeoff (state_coverage)
        state Takeoff;
        
        // DO-331 Coverage Point: CP_STATE_climb (state_coverage)
        state Climb;
        
        // DO-331 Coverage Point: CP_STATE_cruise (state_coverage)
        state Cruise;
        
        // DO-331 Coverage Point: CP_STATE_descent (state_coverage)
        state Descent;
        
        // DO-331 Coverage Point: CP_STATE_landing (state_coverage)
        state Landing;
        
        // DO-331 Coverage Point: CP_STATE_landed (state_coverage)
        state Landed;

        // DO-331 Coverage Point: CP_TRANS_1 (transition_coverage)
        transition Ground -> Takeoff {
            trigger when start_engines;
        }

        // DO-331 Coverage Point: CP_TRANS_2 (transition_coverage)
        transition Takeoff -> Climb {
            if altitude > 1000.0;
        }

        // DO-331 Coverage Point: CP_TRANS_3 (transition_coverage)
        transition Climb -> Cruise {
            if altitude > 30000.0;
        }

        // DO-331 Coverage Point: CP_TRANS_4 (transition_coverage)
        transition Cruise -> Descent {
            trigger when begin_descent;
        }

        // DO-331 Coverage Point: CP_TRANS_5 (transition_coverage)
        transition Descent -> Landing {
            if altitude < 2000.0;
        }

        // DO-331 Coverage Point: CP_TRANS_6 (transition_coverage)
        transition Landing -> Landed {
            if altitude == 0.0;
        }
    }

    requirement def REQ001 {
        doc /* System shall monitor altitude continuously */
    }

    requirement def REQ002 {
        doc /* System shall not exceed 45000 feet altitude */
    }

    satisfy REQ001 by CheckAltitude;
    satisfy REQ001 by GetStatus;
    satisfy REQ002 by CheckAltitude;
}
