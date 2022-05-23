import java.util.*;

public class Experiment2 {
    public static class Lab {
        public String name; // 实验展示名字
        public String exp_name; // 实验运行名字
        public Map<String, Double> result; // label -> ap
        public double fix_mAP;

        public Lab(String name, String exp_name, Map<String, Double> result, double fix_mAP) {
            this.name = name;
            this.exp_name = exp_name;
            this.result = result;
            this.fix_mAP = fix_mAP;
            System.out.printf("mAP: %f\n", this.compute_mAP(null, 0));
        }

        public double compute_mAP(List<String> labelList, int end) {
            if (fix_mAP > 0) {
                return fix_mAP;
            }
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
        for (String key : labList.get(labList.size() - 1).result.keySet()) {
            labelList.add(key);
            result.put(key, 0.0);
        }
        for (Lab lab : labList) {
            if (!lab.exp_name.startsWith("mvpf_self") && !lab.exp_name.startsWith("mvpf_equi_stereo_deform")) {
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
//        labList.add(
//                new Lab("Baseline DPM", "Baseline DPM",
//                        new HashMap<String, Double>() {{
//                        }},
//                        0.294));
//        labList.add(
//                new Lab("Baseline Deng", "Baseline Deng",
//                        new HashMap<String, Double>() {{
//                        }},
//                        0.687));
//        labList.add(
//                new Lab("Baseline PanoBlitzNet EquiConvs", "Baseline PanoBlitzNet EquiConvs",
//                        new HashMap<String, Double>() {{
//                        }},
//                        0.778));
//        labList.add(
//                new Lab("Baseline PanoBlitzNet StdConvs", "Baseline PanoBlitzNet StdConvs",
//                        new HashMap<String, Double>() {{
//                            put("bed", 0.949);
//                            put("painting", 0.850);
//                            put("table", 0.833);
//                            put("mirror", 0.719);
//                            put("window", 0.722);
//                            put("curtain", 0.722);
//                            put("chair", 0.719);
//                            put("light", 0.350);
//                            put("sofa", 0.893);
//                            put("door", 0.755);
//                            put("cabinet", 0.579);
//                            put("bedside", 0.879);
//                            put("tv", 0.911);
//                            put("shelf", 0.305);
//                        }},
//                        0.768));
        labList.add(new Lab("multi_proj", "multi_proj", new HashMap<String, Double>() {{
            put("bed", 0.172);
            put("painting", 0.807);
            put("table", 0.539);
            put("mirror", 0.722);
            put("window", 0.769);
            put("curtain", 0.734);
            put("chair", 0.572);
            put("light", 0.399);
            put("sofa", 0.575);
            put("door", 0.667);
            put("cabinet", 0.475);
            put("bedside", 0.782);
            put("tv", 0.875);
            put("computer", 0.000);
            put("rug", 0.103);
            put("shelf", 0.000);
        }}, -1));
        labList.add(new Lab("multi_proj_nms", "multi_proj_nms", new HashMap<String, Double>() {{
            put("bed", 0.749);
            put("painting", 0.925);
            put("table", 0.671);
            put("mirror", 0.851);
            put("window", 0.930);
            put("curtain", 0.830);
            put("chair", 0.555);
            put("light", 0.449);
            put("sofa", 0.662);
            put("door", 0.773);
            put("cabinet", 0.517);
            put("bedside", 0.854);
            put("tv", 0.873);
            put("computer", 0.000);
            put("rug", 0.491);
            put("shelf", 0.000);
        }}, -1));
        labList.add(new Lab("multi_proj_gen", "multi_proj_gen", new HashMap<String, Double>() {{
            put("bed", 0.133);
            put("painting", 0.875);
            put("table", 0.572);
            put("mirror", 0.683);
            put("window", 0.774);
            put("curtain", 0.741);
            put("chair", 0.567);
            put("light", 0.503);
            put("sofa", 0.457);
            put("door", 0.689);
            put("cabinet", 0.472);
            put("bedside", 0.810);
            put("tv", 0.850);
            put("computer", 0.000);
            put("rug", 0.198);
            put("shelf", 0.000);
        }}, -1));
        labList.add(new Lab("multi_proj_gen_weighted", "multi_proj_gen_weighted", new HashMap<String, Double>() {{
            put("bed", 0.133);
            put("painting", 0.875);
            put("table", 0.572);
            put("mirror", 0.683);
            put("window", 0.774);
            put("curtain", 0.741);
            put("chair", 0.567);
            put("light", 0.503);
            put("sofa", 0.457);
            put("door", 0.689);
            put("cabinet", 0.472);
            put("bedside", 0.810);
            put("tv", 0.850);
            put("computer", 0.000);
            put("rug", 0.198);
            put("shelf", 0.000);
        }}, -1));
        labList.add(new Lab("multi_proj_gen", "multi_proj_gen", new HashMap<String, Double>() {{
            put("bed", 0.778);
            put("painting", 0.799);
            put("table", 0.513);
            put("mirror", 0.762);
            put("window", 0.756);
            put("curtain", 0.839);
            put("chair", 0.607);
            put("light", 0.565);
            put("sofa", 0.598);
            put("door", 0.556);
            put("cabinet", 0.370);
            put("bedside", 0.754);
            put("tv", 0.909);
            put("computer", 0.000);
            put("rug", 0.418);
            put("shelf", 0.995);
        }}, -1));
        labList.add(new Lab("mvpf_self", "mvpf_self", new HashMap<String, Double>() {{
            put("bed", 0.870);
            put("painting", 0.876);
            put("table", 0.637);
            put("mirror", 0.792);
            put("window", 0.810);
            put("curtain", 0.792);
            put("chair", 0.547);
            put("light", 0.547);
            put("sofa", 0.672);
            put("door", 0.618);
            put("cabinet", 0.456);
            put("bedside", 0.799);
            put("tv", 0.955);
            put("computer", 0.000);
            put("rug", 0.995);
            put("shelf", 0.497);
        }}, -1));
        labList.add(new Lab("mvpf_std_std", "mvpf_std_std", new HashMap<String, Double>() {{
            put("bed", 0.703);
            put("painting", 0.922);
            put("table", 0.546);
            put("mirror", 0.749);
            put("window", 0.750);
            put("curtain", 0.695);
            put("chair", 0.362);
            put("light", 0.561);
            put("sofa", 0.648);
            put("door", 0.685);
            put("cabinet", 0.504);
            put("bedside", 0.832);
            put("tv", 0.924);
            put("computer", 0.000);
            put("rug", 0.732);
            put("shelf", 0.000);
        }}, -1));
        labList.add(new Lab("mvpf_equi_std", "mvpf_equi_std", new HashMap<String, Double>() {{
            put("bed", 0.814);
            put("painting", 0.919);
            put("table", 0.589);
            put("mirror", 0.782);
            put("window", 0.800);
            put("curtain", 0.674);
            put("chair", 0.400);
            put("light", 0.524);
            put("sofa", 0.664);
            put("door", 0.640);
            put("cabinet", 0.543);
            put("bedside", 0.804);
            put("tv", 0.936);
            put("computer", 0.000);
            put("rug", 0.732);
        }}, -1));
        labList.add(new Lab("mvpf_equi_stereo", "mvpf_equi_stereo", new HashMap<String, Double>() {{
            put("bed", 0.806);
            put("painting", 0.931);
            put("table", 0.589);
            put("mirror", 0.733);
            put("window", 0.819);
            put("curtain", 0.624);
            put("chair", 0.372);
            put("light", 0.472);
            put("sofa", 0.567);
            put("door", 0.597);
            put("cabinet", 0.478);
            put("bedside", 0.804);
            put("tv", 0.933);
            put("computer", 0.000);
            put("rug", 0.527);
        }}, -1));
        labList.add(new Lab("mvpf_equi_stereo_2", "mvpf_equi_stereo_2", new HashMap<String, Double>() {{
            put("bed", 0.807);
            put("painting", 0.876);
            put("table", 0.662);
            put("mirror", 0.773);
            put("window", 0.864);
            put("curtain", 0.670);
            put("chair", 0.466);
            put("light", 0.422);
            put("sofa", 0.656);
            put("door", 0.533);
            put("cabinet", 0.452);
            put("bedside", 0.832);
            put("tv", 0.923);
            put("computer", 0.000);
            put("rug", 0.721);
        }}, -1));
        labList.add(new Lab("mvpf_equi_stereo_deform", "mvpf_equi_stereo_deform", new HashMap<String, Double>() {{
            put("bed", 0.810);
            put("painting", 0.904);
            put("table", 0.522);
            put("mirror", 0.768);
            put("window", 0.797);
            put("curtain", 0.846);
            put("chair", 0.626);
            put("light", 0.587);
            put("sofa", 0.661);
            put("door", 0.592);
            put("cabinet", 0.415);
            put("bedside", 0.792);
            put("tv", 0.955);
            put("computer", 0.000);
            put("rug", 0.418);
            put("shelf", 0.995);
        }}, -1));
        labList.add(new Lab("mvpf_detect", "mvpf_detect", new HashMap<String, Double>() {{
            put("bed", 0.778);
            put("painting", 0.799);
            put("table", 0.513);
            put("mirror", 0.762);
            put("window", 0.756);
            put("curtain", 0.839);
            put("chair", 0.607);
            put("light", 0.565);
            put("sofa", 0.598);
            put("door", 0.556);
            put("cabinet", 0.370);
            put("bedside", 0.754);
            put("tv", 0.909);
            put("computer", 0.000);
            put("rug", 0.418);
            put("shelf", 0.995);
        }}, -1));
        List<String> labelList = SortLabel_mAP(labList);
        for (int i = 2; i < 8; i++) {
            Sort_mAP(labList, labelList, i, false);
        }
        Sort_mAP(labList, labelList, 7, true);
    }
}