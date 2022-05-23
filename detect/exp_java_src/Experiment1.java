import java.util.*;

public class Experiment1 {
    public static class Lab {
        public String name;
        public String exp_name;
        public Map<String, Double> result; // label -> ap

        public Lab(String name, String exp_name, Map<String, Double> result) {
            this.name = name;
            this.exp_name = exp_name;
            this.result = result;
            System.out.printf("mAP: %f\n", this.compute_mAP(null, 0));
        }

        public double compute_mAP(List<String> labelList, int end) {
            double mAP = 0.0;
            if (labelList == null) {
                for (String key : this.result.keySet()) {
                    mAP += result.get(key);
                }
                return mAP / result.size();
            }
            if (end <= 0) {
                end = labelList.size();
            }
            for (int i = 0; i < end; i++) {
                mAP += result.get(labelList.get(i));
            }
            return mAP / end;
        }
    }

    public static List<String> SortLabel_mAP(List<Lab> labList) {
        Map<String, Double> result = new HashMap<>();
        List<String> labelList = new ArrayList<>();
        for (String key : labList.get(0).result.keySet()) {
            labelList.add(key);
            result.put(key, 0.0);
        }
        for (Lab lab : labList) {
            if (!lab.exp_name.startsWith("stereoconv_1_deform_bak")) {
                continue;
            }
            for (String key : labelList) {
                result.put(key, result.get(key) + lab.result.get(key));
            }
        }
        labelList.sort(Comparator.comparingDouble(result::get));
        Collections.reverse(labelList);
        for (String key : labelList) {
            System.out.printf("%s: %f\n", key, result.get(key));
        }
        return labelList;
        /**
         * glass: 0.321000
         * shelf: 2.892000
         * light: 4.005000
         * computer: 4.158000
         * rug: 5.561000
         * cabinet: 6.003000
         * chair: 7.536000
         * window: 7.682000
         * mirror: 7.746000
         * door: 8.502000
         * table: 9.294000
         * bedside: 9.475000
         * curtain: 9.965000
         * sofa: 10.542000
         * painting: 11.245000
         * tv: 11.418000
         * bed: 11.514000
         */
    }

    public static void Sort_mAP(List<Lab> labList, List<String> labelList, int end, boolean out) {
        Map<String, Double> result = new HashMap<>();
        List<String> labNameList = new ArrayList<>();
        for (Lab lab : labList) {
            labNameList.add(lab.name);
            result.put(lab.name, lab.compute_mAP(labelList, end));
        }
        labNameList.sort(Comparator.comparingDouble(result::get));
        Collections.reverse(labNameList);
        System.out.printf("%d==========\n", end);
        for (String key : labNameList) {
            System.out.printf("%s: %f\n", key, result.get(key));
        }
        if (out) {
            StringBuilder buf = new StringBuilder();
            buf.append("model");
            buf.append(" & ");
            buf.append("mAP");
            for (int i = 0; i < end; i++) {
                buf.append(" & ");
                buf.append(labelList.get(i));
            }
            buf.append(" \\\\");
            System.out.println(buf);
            for (Lab lab : labList) {
                buf = new StringBuilder();
                buf.append(lab.name);
                buf.append(" & ");
                buf.append(String.format("%.1f", lab.compute_mAP(labelList, end) * 100));
                for (int i = 0; i < end; i++) {
                    buf.append(" & ");
                    buf.append(String.format("%.1f", lab.result.get(labelList.get(i)) * 100));
                }
                buf.append(" \\\\");
                System.out.println(buf);
            }

        }
    }

    public static void main(String[] args) {
        List<Lab> labList = new ArrayList<>();
        labList.add(new Lab("Baseline", "stdconv", new HashMap<String, Double>() {{
            put("bed", 0.881);
            put("painting", 0.875);
            put("table", 0.710);
            put("mirror", 0.599);
            put("window", 0.595);
            put("curtain", 0.776);
            put("chair", 0.585);
            put("light", 0.318);
            put("sofa", 0.793);
            put("door", 0.648);
            put("cabinet", 0.474);
            put("bedside", 0.747);
            put("tv", 0.868);
            put("computer", 0.034);
            put("glass", 0.036);
            put("rug", 0.456);
            put("shelf", 0.272);
        }}));
        labList.add(new Lab("DeformConv-Backbone-Layer1", "deformconv_1", new HashMap<String, Double>() {{
            put("bed", 0.860);
            put("painting", 0.870);
            put("table", 0.704);
            put("mirror", 0.606);
            put("window", 0.589);
            put("curtain", 0.784);
            put("chair", 0.562);
            put("light", 0.342);
            put("sofa", 0.812);
            put("door", 0.647);
            put("cabinet", 0.484);
            put("bedside", 0.747);
            put("tv", 0.890);
            put("computer", 0.141);
            put("glass", 0.072);
            put("rug", 0.462);
            put("shelf", 0.184);
        }}));
        labList.add(new Lab("DeformConv-Backbone", "deformconv_minback", new HashMap<String, Double>() {{
            put("bed", 0.873);
            put("painting", 0.883);
            put("table", 0.736);
            put("mirror", 0.612);
            put("window", 0.608);
            put("curtain", 0.782);
            put("chair", 0.580);
            put("light", 0.338);
            put("sofa", 0.822);
            put("door", 0.692);
            put("cabinet", 0.488);
            put("bedside", 0.785);
            put("tv", 0.868);
            put("computer", 0.146);
            put("glass", 0.035);
            put("rug", 0.393);
            put("shelf", 0.266);
        }}));
        labList.add(new Lab("DeformConv-Full", "deformconv_all", new HashMap<String, Double>() {{
            put("bed", 0.906);
            put("painting", 0.885);
            put("table", 0.733);
            put("mirror", 0.603);
            put("window", 0.625);
            put("curtain", 0.789);
            put("chair", 0.577);
            put("light", 0.372);
            put("sofa", 0.834);
            put("door", 0.698);
            put("cabinet", 0.490);
            put("bedside", 0.749);
            put("tv", 0.905);
            put("computer", 0.309);
            put("glass", 0.014);
            put("rug", 0.424);
            put("shelf", 0.214);
        }}));
        labList.add(new Lab("StereoConv-Backbone-Layer5", "stereoconv_1_5", new HashMap<String, Double>() {{
            put("bed", 0.900);
            put("painting", 0.888);
            put("table", 0.720);
            put("mirror", 0.608);
            put("window", 0.575);
            put("curtain", 0.775);
            put("chair", 0.587);
            put("light", 0.297);
            put("sofa", 0.806);
            put("door", 0.654);
            put("cabinet", 0.468);
            put("bedside", 0.768);
            put("tv", 0.872);
            put("computer", 0.427);
            put("glass", 0.071);
            put("rug", 0.438);
            put("shelf", 0.241);
        }}));
        labList.add(new Lab("StereoConv-Backbone-Layer3", "stereoconv_3_3", new HashMap<String, Double>() {{
            put("bed", 0.878);
            put("painting", 0.856);
            put("table", 0.714);
            put("mirror", 0.594);
            put("window", 0.559);
            put("curtain", 0.747);
            put("chair", 0.608);
            put("light", 0.299);
            put("sofa", 0.827);
            put("door", 0.662);
            put("cabinet", 0.474);
            put("bedside", 0.728);
            put("tv", 0.866);
            put("computer", 0.488);
            put("glass", 0.075);
            put("rug", 0.406);
            put("shelf", 0.222);
        }}));
        labList.add(new Lab("StereoConv-Backbone-Layer1", "stereoconv_4_1", new HashMap<String, Double>() {{
            put("bed", 0.901);
            put("painting", 0.871);
            put("table", 0.732);
            put("mirror", 0.577);
            put("window", 0.585);
            put("curtain", 0.762);
            put("chair", 0.595);
            put("light", 0.305);
            put("sofa", 0.825);
            put("door", 0.652);
            put("cabinet", 0.499);
            put("bedside", 0.759);
            put("tv", 0.863);
            put("computer", 0.319);
            put("glass", 0.014);
            put("rug", 0.412);
            put("shelf", 0.217);
        }}));
        labList.add(new Lab("StereoConv-Backbone-Layer2", "stereoconv_8_2", new HashMap<String, Double>() {{
            put("bed", 0.885);
            put("painting", 0.851);
            put("table", 0.697);
            put("mirror", 0.571);
            put("window", 0.568);
            put("curtain", 0.738);
            put("chair", 0.595);
            put("light", 0.270);
            put("sofa", 0.803);
            put("door", 0.626);
            put("cabinet", 0.441);
            put("bedside", 0.685);
            put("tv", 0.864);
            put("computer", 0.173);
            put("glass", 0.009);
            put("rug", 0.452);
            put("shelf", 0.174);
        }}));
        labList.add(new Lab("StereoConv-Backbone-Layer4", "stereoconv_9_4", new HashMap<String, Double>() {{
            put("bed", 0.873);
            put("painting", 0.847);
            put("table", 0.712);
            put("mirror", 0.605);
            put("window", 0.590);
            put("curtain", 0.775);
            put("chair", 0.555);
            put("light", 0.281);
            put("sofa", 0.788);
            put("door", 0.643);
            put("cabinet", 0.433);
            put("bedside", 0.695);
            put("tv", 0.873);
            put("computer", 0.547);
            put("glass", 0.005);
            put("rug", 0.386);
            put("shelf", 0.201);
        }}));
        labList.add(new Lab("StereoConv-Backbone-Layer6", "stereoconv_10_6", new HashMap<String, Double>() {{
            put("bed", 0.875);
            put("painting", 0.850);
            put("table", 0.691);
            put("mirror", 0.615);
            put("window", 0.560);
            put("curtain", 0.772);
            put("chair", 0.578);
            put("light", 0.312);
            put("sofa", 0.804);
            put("door", 0.630);
            put("cabinet", 0.441);
            put("bedside", 0.722);
            put("tv", 0.871);
            put("computer", 0.164);
            put("glass", 0.005);
            put("rug", 0.428);
            put("shelf", 0.194);
        }}));
        labList.add(new Lab("StereoConv-Head", "stereoconv_12_head", new HashMap<String, Double>() {{
            put("bed", 0.887);
            put("painting", 0.831);
            put("table", 0.694);
            put("mirror", 0.580);
            put("window", 0.594);
            put("curtain", 0.759);
            put("chair", 0.550);
            put("light", 0.271);
            put("sofa", 0.801);
            put("door", 0.621);
            put("cabinet", 0.430);
            put("bedside", 0.715);
            put("tv", 0.856);
            put("computer", 0.195);
            put("glass", 0.004);
            put("rug", 0.429);
            put("shelf", 0.268);
        }}));
        labList.add(new Lab("StereoConv-Backbone", "stereoconv_minback", new HashMap<String, Double>() {{
            put("bed", 0.859);
            put("painting", 0.852);
            put("table", 0.708);
            put("mirror", 0.592);
            put("window", 0.594);
            put("curtain", 0.751);
            put("chair", 0.565);
            put("light", 0.299);
            put("sofa", 0.795);
            put("door", 0.656);
            put("cabinet", 0.421);
            put("bedside", 0.704);
            put("tv", 0.884);
            put("computer", 0.516);
            put("glass", 0.004);
            put("rug", 0.438);
            put("shelf", 0.181);
        }}));
        labList.add(new Lab("DeformConv-Full + StereoConv-Backbone-Layer5", "stereoconv_1_deform", new HashMap<String, Double>() {{
            put("bed", 0.904);
            put("painting", 0.875);
            put("table", 0.713);
            put("mirror", 0.581);
            put("window", 0.624);
            put("curtain", 0.753);
            put("chair", 0.575);
            put("light", 0.315);
            put("sofa", 0.818);
            put("door", 0.685);
            put("cabinet", 0.464);
            put("bedside", 0.735);
            put("tv", 0.915);
            put("computer", 0.527);
            put("glass", 0.007);
            put("rug", 0.422);
            put("shelf", 0.215);
        }}));
        labList.add(new Lab("DeformConv-Backbone + StereoConv-Backbone-Layer5", "stereoconv_1_deform_bak", new HashMap<String, Double>() {{
            put("bed", 0.905);
            put("painting", 0.894);
            put("table", 0.766);
            put("mirror", 0.615);
            put("window", 0.624);
            put("curtain", 0.784);
            put("chair", 0.604);
            put("light", 0.324);
            put("sofa", 0.836);
            put("door", 0.680);
            put("cabinet", 0.484);
            put("bedside", 0.721);
            put("tv", 0.891);
            put("computer", 0.318);
            put("glass", 0.005);
            put("rug", 0.408);
            put("shelf", 0.309);
        }}));
        List<String> labelList = SortLabel_mAP(labList);
        for (int i = 2; i < 12; i++) {
            Sort_mAP(labList, labelList, i, false);
        }
        Sort_mAP(labList, labelList, 6, true);
    }
}

/**
 * painting: 0.894000
 * tv: 0.891000
 * sofa: 0.836000
 * curtain: 0.784000
 * table: 0.766000
 * bedside: 0.721000
 * <p>
 * 6==========
 * stereoconv_1_deform_bak: 0.846000
 * deformconv_all: 0.842000
 * stereoconv_1_deform: 0.829667
 * stereoconv_1: 0.826833
 * stereoconv_4: 0.825667
 * deformconv_1: 0.820000
 * stdconv: 0.817167
 * stereoconv_3: 0.814667
 * stereoconv_9: 0.811333
 * stereoconv_10: 0.810500
 * stereoconv_minback: 0.808167
 * stereoconv_8: 0.806333
 * stereoconv_12_head: 0.804667
 */
